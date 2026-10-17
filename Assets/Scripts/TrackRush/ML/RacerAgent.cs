using System.Collections.Generic;
using Unity.InferenceEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using UnityEngine;
using UnityEngine.InputSystem;
using Random = UnityEngine.Random;

namespace TrackRush
{
    /// RL driver for Track Rush. Sees its own motion and the road ahead in its own frame, and
    /// outputs [steer, pedal] where pedal > 0 is throttle and pedal < 0 is brake.
    ///
    /// Training == true (set by TrainingManager): the agent tracks its own progress, earns rewards
    /// and respawns itself. Training == false: it just drives (e.g. as the AI rival in the game).
    [RequireComponent(typeof(CarController))]
    public class RacerAgent : Agent
    {
        public const string Behavior = "TMRacer";
        public const int ObservationSize = 44;
        public const int ActionSize = 2;
        public const int DecisionPeriod = 5; // physics runs at 60 Hz -> 12 decisions/s

        static readonly float[] lookAhead = { 10f, 20f, 35f, 50f, 75f, 100f, 140f };

        // reward shaping
        const float ProgressReward = 0.02f;   // per metre along the track
        const float CheckpointReward = 1f;
        const float FinishReward = 5f;
        const float GrassPenalty = 0.01f;     // per physics step
        const float WallHitPenalty = 0.3f;    // once, on impact
        const float WallScrapePenalty = 0.02f;
        const float FailPenalty = 1f;         // fell off, stuck or wrong way

        public struct EpisodeStats
        {
            public float Reward;
            public bool Finished;
            public bool FromStart;    // started on the start line, so FinishTime is a full run
            public float FinishTime;  // seconds of game time; valid when Finished
        }

        /// Raised when a training episode ends, for spectator stats.
        public static event System.Action<RacerAgent, EpisodeStats> EpisodeEnded;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        static void ResetStatics() => EpisodeEnded = null;

        public bool Training;
        /// Training only: start every episode on the start line instead of a random safe spot.
        public bool MassStart;

        CarController car;
        float prevSteer, prevPedal, stuckTime, wrongWayTime, lostTime;
        int nextCheckpoint, episodeSteps;
        bool finished, fromStart;
        float lastReward;
        static readonly List<int> spawnBuffer = new();

        /// Adds the ML-Agents components to a car. The GameObject is deactivated while doing so,
        /// because the Agent reads its BehaviorParameters as soon as it is enabled.
        public static RacerAgent Attach(CarController car, bool training, BehaviorType type, ModelAsset model = null)
        {
            var go = car.gameObject;
            bool wasActive = go.activeSelf;
            go.SetActive(false);

            var bp = go.AddComponent<BehaviorParameters>();
            bp.BehaviorName = Behavior;
            bp.BrainParameters.VectorObservationSize = ObservationSize;
            bp.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(ActionSize);
            bp.BehaviorType = type;
            bp.Model = model;
            bp.DeterministicInference = !training;

            var agent = go.AddComponent<RacerAgent>(); // before DecisionRequester, which requires an Agent
            agent.Training = training;
            agent.MaxStep = training ? 6000 : 0;       // 100 s episodes while training

            var dr = go.AddComponent<DecisionRequester>();
            dr.DecisionPeriod = DecisionPeriod;
            dr.TakeActionsBetweenDecisions = true;

            go.SetActive(wasActive);
            return agent;
        }

        public override void Initialize()
        {
            car = GetComponent<CarController>();
        }

        public override void OnEpisodeBegin()
        {
            if (Training && episodeSteps > 0)
                EpisodeEnded?.Invoke(this, new EpisodeStats
                {
                    Reward = lastReward, Finished = finished, FromStart = fromStart,
                    FinishTime = episodeSteps * Time.fixedDeltaTime,
                });
            episodeSteps = 0;
            finished = false;
            lastReward = 0;
            prevSteer = prevPedal = stuckTime = wrongWayTime = lostTime = 0;
            if (!Training) return;

            var t = car.Track;
            int index = t.StartIndex;
            float speed = 0f;
            if (!MassStart && Random.value > 0.3f)
            {
                index = RandomSafeIndex(t);
                speed = Random.Range(0.3f, 0.9f) * Mathf.Min(t.TargetSpeed[index], CarController.TopSpeed);
            }
            car.Frozen = false;
            fromStart = index == t.StartIndex && speed == 0f;
            car.PlaceOnTrack(index, speed);
            nextCheckpoint = 0;
            while (nextCheckpoint < t.Checkpoints.Length && t.Checkpoints[nextCheckpoint] <= index) nextCheckpoint++;
        }

        /// A flat, upright, on-road sample before the finish (no gaps, loops or steep banks).
        static int RandomSafeIndex(Track t)
        {
            for (int tries = 0; tries < 50; tries++)
            {
                int i = Random.Range(0, t.FinishIndex);
                if (!t.IsGap[i] && !t.IsLoop[i] && t.Up[i].y > 0.9f) return i;
            }
            return t.StartIndex;
        }

        public override void CollectObservations(VectorSensor s)
        {
            var t = car.Track;
            if (t == null || car.TrackIndex < 0)
            {
                for (int k = 0; k < ObservationSize; k++) s.AddObservation(0f);
                return;
            }
            var rot = car.Rotation;
            var inv = Quaternion.Inverse(rot);
            int i = car.TrackIndex;
            Vector3 fwd = rot * Vector3.forward, up = rot * Vector3.up;

            // own motion (4)
            var v = inv * car.Velocity;
            s.AddObservation(v.x / 20f);
            s.AddObservation(v.y / 20f);
            s.AddObservation(v.z / CarController.BoostTopSpeed);
            s.AddObservation(car.YawRate / 3f);

            // pose relative to the road (6)
            float heading = Vector3.SignedAngle(t.Forward[i], Vector3.ProjectOnPlane(fwd, t.Up[i]), t.Up[i]) * Mathf.Deg2Rad;
            s.AddObservation(Mathf.Sin(heading));
            s.AddObservation(Mathf.Cos(heading));
            s.AddObservation(Vector3.Dot(up, t.Right[i]));     // roll relative to the road
            s.AddObservation(Vector3.Dot(fwd, t.Up[i]));       // pitch relative to the road
            s.AddObservation(car.LateralOffset / t.HalfWidth);
            s.AddObservation(Mathf.Clamp(car.HeightAboveTrack / 5f, -3f, 3f));

            // contact state (3)
            s.AddObservation(car.Grounded ? 1f : 0f);
            s.AddObservation(car.OnGrass ? 1f : 0f);
            s.AddObservation(car.BoostTime / car.BoostDuration);

            // road ahead in the car's frame, plus whether that stretch is a gap (28)
            foreach (float d in lookAhead)
            {
                t.SampleAt(car.Progress + d, out var p, out _, out _);
                var local = inv * (p - car.Position);
                s.AddObservation(local.x / 30f);
                s.AddObservation(local.y / 30f);
                s.AddObservation(local.z / d);
                s.AddObservation(t.IsGap[t.IndexAt(car.Progress + d)] ? 1f : 0f);
            }

            // progress through the track (1) and previous action (2)
            s.AddObservation(car.Progress / t.Length);
            s.AddObservation(prevSteer);
            s.AddObservation(prevPedal);
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            var a = actions.ContinuousActions;
            float steer = Mathf.Clamp(a[0], -1f, 1f), pedal = Mathf.Clamp(a[1], -1f, 1f);
            car.Controls = new CarInput { Steer = steer, Throttle = Mathf.Max(pedal, 0f), Brake = Mathf.Max(-pedal, 0f) };
            prevSteer = steer;
            prevPedal = pedal;
            if (!Training) return;

            var t = car.Track;
            episodeSteps++;
            int prev = car.TrackIndex;
            AddReward(t.Locate(car) * ProgressReward);
            if (car.OnGrass) AddReward(-GrassPenalty);
            if (car.WallImpactSpeed > 1f) AddReward(-WallHitPenalty);
            if (car.TouchingWall) AddReward(-WallScrapePenalty);

            int cur = car.TrackIndex;
            if (cur - prev < 60 && t.InGate(car))
            {
                if (nextCheckpoint < t.Checkpoints.Length && prev < t.Checkpoints[nextCheckpoint] && cur >= t.Checkpoints[nextCheckpoint])
                {
                    nextCheckpoint++;
                    AddReward(CheckpointReward);
                }
                else if (nextCheckpoint >= t.Checkpoints.Length && prev < t.FinishIndex && cur >= t.FinishIndex)
                {
                    AddReward(FinishReward);
                    finished = true;
                    lastReward = GetCumulativeReward();
                    EndEpisode();
                    return;
                }
            }

            float dt = Time.fixedDeltaTime;
            float heading = Vector3.Angle(t.Forward[cur], car.Rotation * Vector3.forward);
            stuckTime = car.ForwardSpeed < 2f ? stuckTime + dt : 0f;
            wrongWayTime = car.Grounded && heading > 100f ? wrongWayTime + dt : 0f;
            lostTime = !car.OnRoad ? lostTime + dt : 0f;
            if (stuckTime > 4f || wrongWayTime > 1.5f || lostTime > 3f)
            {
                AddReward(-FailPenalty);
                lastReward = GetCumulativeReward();
                EndEpisode();
                return;
            }
            lastReward = GetCumulativeReward();
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var a = actionsOut.ContinuousActions;
            float steer = 0, pedal = 0;
            var kb = Keyboard.current;
            if (kb != null)
            {
                if (kb.aKey.isPressed || kb.leftArrowKey.isPressed) steer -= 1;
                if (kb.dKey.isPressed || kb.rightArrowKey.isPressed) steer += 1;
                if (kb.wKey.isPressed || kb.upArrowKey.isPressed) pedal += 1;
                if (kb.sKey.isPressed || kb.downArrowKey.isPressed) pedal -= 1;
            }
            var gp = Gamepad.current;
            if (gp != null)
            {
                float sx = gp.leftStick.x.ReadValue();
                if (Mathf.Abs(sx) > 0.1f) steer = sx;
                float p = gp.rightTrigger.ReadValue() - gp.leftTrigger.ReadValue();
                if (Mathf.Abs(p) > 0.05f) pedal = p;
            }
            a[0] = steer;
            a[1] = pedal;
        }
    }
}

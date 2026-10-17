using System.Collections.Generic;
using UnityEngine;

namespace TrackRush
{
    public struct CarInput
    {
        public float Throttle, Brake, Steer;
    }

    /// Arcade stunt car. Four ground probes find the surface; while grounded the car is glued
    /// to it (so it follows banks and loops), and its velocity is split into forward / sideways
    /// parts for engine, brakes and grip. In the air it is a plain ballistic body.
    [RequireComponent(typeof(Rigidbody))]
    public class CarController : MonoBehaviour
    {
        public const float TopSpeed = 75f;        // m/s (270 km/h) on engine power alone
        public const float BoostTopSpeed = 105f;  // m/s (378 km/h) while boosting
        public const float Gravity = TrackBuilder.Gravity;
        const float RideHeight = 0.9f;            // probe origin height above the road at rest
        const float ProbeLength = 1.9f;

        /// Every active car, for agents that observe their neighbours.
        public static readonly List<CarController> All = new();

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        static void ResetStatics() => All.Clear();

        public string DriverName, Abbrev;
        public Team Team;
        public bool IsPlayer;

        public float MaxSpeed = TopSpeed;
        public float EngineAccel = 16f;
        public float BoostAccel = 32f;
        public float BrakeDecel = 30f;
        public float Grip = 45f;          // max sideways acceleration, m/s²
        public float LateralGrip = 70f;   // how quickly sideways slip is killed
        public float BoostDuration = 1.6f;

        public CarInput Controls;
        public bool Frozen = true;
        public Track Track;

        // written by Track.Locate
        public int TrackIndex = -1;
        public float Progress, LateralOffset, HeightAboveTrack;
        public bool OnRoad;

        public bool Grounded { get; private set; }
        public bool OnGrass { get; private set; }
        public float BoostTime { get; private set; }
        public float ForwardSpeed { get; private set; }
        public float LateralSpeed { get; private set; }
        public float YawRate { get; private set; }
        public float SpeedKmh => Mathf.Abs(ForwardSpeed) * 3.6f;
        public int Gear { get; private set; } = 1;
        public float RpmFraction { get; private set; }

        static readonly Vector3[] probes =
        {
            new(-0.75f, RideHeight, 1.5f), new(0.75f, RideHeight, 1.5f),
            new(-0.75f, RideHeight, -1.4f), new(0.75f, RideHeight, -1.4f),
        };

        Rigidbody rb;
        Transform[] steerPivots, spinNodes;
        float wheelAngle, steerVisual, detachTimer;
        Vector3 velocity;
        bool insideWalls = true;

        public const int CarLayer = 2; // Ignore Raycast: probes skip cars, and cars never collide with each other
        const float WallClearance = 1.0f; // half the car's width

        /// Cars pass through each other (Trackmania style). One layer rule instead of per-pair ignores,
        /// which would be ~500k calls for 1000 cars.
        public static void ConfigureCarLayer() => Physics.IgnoreLayerCollision(CarLayer, CarLayer, true);

        // physics pose (the Transform is interpolated for rendering, so logic reads these)
        public Vector3 Position => rb.position;
        public Quaternion Rotation => rb.rotation;
        public Vector3 Velocity => velocity;
        public bool TouchingWall { get; private set; }
        /// Speed into the wall on the step the car hit it, else 0.
        public float WallImpactSpeed { get; private set; }

        void Awake() => rb = GetComponent<Rigidbody>();
        void OnEnable() => All.Add(this);
        void OnDisable() => All.Remove(this);

        public void Setup(string driver, string abbrev, Team team, bool isPlayer, Transform[] steer, Transform[] spin)
        {
            DriverName = driver;
            Abbrev = abbrev;
            Team = team;
            IsPlayer = isPlayer;
            steerPivots = steer;
            spinNodes = spin;
        }

        public void Place(Pose pose, float speed = 0f)
        {
            transform.SetPositionAndRotation(pose.position, pose.rotation);
            rb.position = pose.position;
            rb.rotation = pose.rotation;
            velocity = pose.rotation * Vector3.forward * speed;
            ForwardSpeed = speed;
            BoostTime = 0;
            detachTimer = 0;
            insideWalls = true;
            TouchingWall = false;
            WallImpactSpeed = 0;
        }

        /// Puts the car on the track at a sample (used for respawns), resetting progress tracking there.
        public void PlaceOnTrack(int index, float speed = 0f)
        {
            Place(Track.PoseAt(index), speed);
            TrackIndex = index;
            Progress = Track.Dist[index];
            Track.Locate(this);
        }

        // The car moves itself (kinematic rigidbody, no collider): four ground probes find the road,
        // and the walls are a clamp against the track edges. With no car colliders in the scene,
        // ground rays only ever test the track, so 1000 overlapping cars stay cheap.
        void FixedUpdate()
        {
            float dt = Time.fixedDeltaTime;
            var c = Frozen ? default : Controls;
            BoostTime = Mathf.Max(0, BoostTime - dt);
            var pos = rb.position;
            var rot = rb.rotation;
            var down = rot * Vector3.down;

            // ---- ground probes
            int hits = 0;
            Vector3 normalSum = Vector3.zero;
            float distSum = 0;
            bool grass = false;
            detachTimer -= dt;
            if (detachTimer <= 0)
                foreach (var p in probes)
                {
                    if (!Physics.Raycast(pos + rot * p, down, out var hit, ProbeLength, Physics.DefaultRaycastLayers, QueryTriggerInteraction.Ignore)) continue;
                    if (!Track.Surfaces.TryGetValue(hit.collider, out bool isRoad)) continue;
                    hits++;
                    normalSum += hit.normal;
                    distSum += hit.distance;
                    if (!isRoad) grass = true;
                }
            Grounded = hits >= 2;
            OnGrass = Grounded && grass;

            if (Track != null && Grounded && !OnGrass && TrackIndex >= 0 && Track.IsBoost[TrackIndex]
                && Mathf.Abs(LateralOffset) < Track.HalfWidth * 0.75f)
                BoostTime = BoostDuration;

            if (Grounded) DriveOnGround(c, (normalSum / hits).normalized, distSum / hits, dt, ref rot);
            else Fly(c, dt, ref rot);

            pos += velocity * dt;
            ApplyWalls(ref pos, dt);
            rb.MovePosition(pos);
            rb.MoveRotation(rot);

            steerVisual = Mathf.Lerp(steerVisual, c.Steer * 22f, 1f - Mathf.Exp(-12f * dt));
        }

        void DriveOnGround(CarInput c, Vector3 n, float dist, float dt, ref Quaternion rot)
        {
            Vector3 f = Vector3.ProjectOnPlane(rot * Vector3.forward, n).normalized;
            Vector3 r = Vector3.Cross(n, f);
            float fs = Vector3.Dot(velocity, f), ls = Vector3.Dot(velocity, r);

            // gravity along the surface: slows climbs and loops, speeds descents
            Vector3 g = Vector3.down * Gravity;
            fs += Vector3.Dot(g, f) * dt;
            ls += Vector3.Dot(g, r) * dt;

            bool boosting = BoostTime > 0;
            float maxV = OnGrass ? 35f : boosting ? BoostTopSpeed : MaxSpeed;
            if (c.Throttle > 0 && fs >= -0.5f)
            {
                float x = Mathf.Clamp01(fs / maxV);
                fs += c.Throttle * EngineAccel * (1f - x * x) * dt;
            }
            else if (c.Throttle > 0)
                fs += c.Throttle * BrakeDecel * dt;
            if (boosting && fs < BoostTopSpeed) fs += BoostAccel * dt;
            if (c.Brake > 0)
            {
                if (fs > 0.5f) fs = Mathf.Max(0, fs - c.Brake * BrakeDecel * dt);
                else fs = Mathf.Max(fs - c.Brake * 10f * dt, -15f);
            }

            float drag = 0.3f + 0.0003f * fs * fs + (OnGrass ? 5f : 0f);
            if (Mathf.Abs(fs) > maxV) drag += 8f;
            fs = Mathf.MoveTowards(fs, 0, drag * dt);

            // bicycle-model yaw, capped by tyre grip
            float speedAbs = Mathf.Abs(fs);
            float steerAngle = 26f * Mathf.Lerp(1f, 0.3f, speedAbs / BoostTopSpeed) * c.Steer;
            float yaw = fs * Mathf.Tan(steerAngle * Mathf.Deg2Rad) / 3.4f;
            float grip = Grip * (OnGrass ? 0.55f : 1f);
            float maxYaw = grip / Mathf.Max(speedAbs, 1f);
            yaw = Mathf.Clamp(yaw, -maxYaw, maxYaw);
            ls = Mathf.MoveTowards(ls, 0, LateralGrip * (OnGrass ? 0.5f : 1f) * dt);

            // hold ride height; this is what glues the car to banks and loops
            float vn = Mathf.Clamp((RideHeight - dist) * 15f, -6f, 6f);

            // too slow upside down: fall off
            if (n.y < -0.2f && fs < 18f) detachTimer = 0.6f;

            velocity = f * fs + r * ls + n * vn;
            var oldUp = rot * Vector3.up;
            var up = Vector3.Slerp(oldUp, n, 1f - Mathf.Exp(-60f * dt));
            rot = Quaternion.AngleAxis(yaw * Mathf.Rad2Deg * dt, up) * (Quaternion.FromToRotation(oldUp, up) * rot);

            ForwardSpeed = fs;
            LateralSpeed = ls;
            YawRate = yaw;
        }

        void Fly(CarInput c, float dt, ref Quaternion rot)
        {
            velocity += Vector3.down * Gravity * dt;
            velocity *= 1f - 0.0002f * velocity.magnitude * dt;

            // gentle air control plus self-levelling so jumps land on the wheels
            var flatV = new Vector3(velocity.x, 0, velocity.z);
            var heading = flatV.sqrMagnitude > 4f ? flatV.normalized : Vector3.ProjectOnPlane(rot * Vector3.forward, Vector3.up).normalized;
            if (heading.sqrMagnitude < 0.5f) heading = Vector3.forward;
            heading = Quaternion.AngleAxis(c.Steer * 40f * dt, Vector3.up) * heading;
            rot = Quaternion.Slerp(rot, Quaternion.LookRotation(heading, Vector3.up), 1f - Mathf.Exp(-2.5f * dt));

            ForwardSpeed = Vector3.Dot(velocity, rot * Vector3.forward);
            LateralSpeed = Vector3.Dot(velocity, rot * Vector3.right);
            YawRate = 0;
        }

        /// Walls as a constraint: a car that was between the walls can't leave through them.
        /// (Cars coming back from the grass or out of the air pass through, which is forgiving.)
        void ApplyWalls(ref Vector3 pos, float dt)
        {
            WallImpactSpeed = 0;
            TouchingWall = false;
            if (Track == null) return;
            int i = Track.FindNearest(pos, TrackIndex); // local lookup only; TrackIndex belongs to Track.Locate
            if (Track.IsGap[i]) { insideWalls = false; return; }
            Vector3 d = pos - Track.Points[i];
            float lat = Vector3.Dot(d, Track.Right[i]), h = Vector3.Dot(d, Track.Up[i]);
            if (h < -1.5f || h > Track.WallHeight + 1.2f) { insideWalls = false; return; }
            float limit = Track.HalfWidth - WallClearance;
            if (Mathf.Abs(lat) <= limit) { insideWalls = true; return; }
            if (!insideWalls) return;

            float side = Mathf.Sign(lat);
            pos -= Track.Right[i] * (lat - side * limit);
            float into = Vector3.Dot(velocity, Track.Right[i]) * side;
            if (into > 0)
            {
                velocity -= Track.Right[i] * (side * into * 1.3f); // small bounce off the wall
                WallImpactSpeed = into;
            }
            velocity *= 1f - 0.8f * dt; // scraping costs speed
            TouchingWall = true;
        }

        void Update()
        {
            float s = Mathf.Abs(ForwardSpeed) / TopSpeed * 7f;
            Gear = Mathf.Clamp(Mathf.FloorToInt(s) + 1, 1, 7);
            RpmFraction = 0.35f + 0.65f * (Gear == 7 ? Mathf.Clamp01(s - 6f) : s - Mathf.Floor(s));

            wheelAngle = (wheelAngle + ForwardSpeed / 0.37f * Mathf.Rad2Deg * Time.deltaTime) % 360f;
            if (spinNodes != null)
                foreach (var w in spinNodes) w.localRotation = Quaternion.Euler(wheelAngle, 0, 0);
            if (steerPivots != null)
                foreach (var p in steerPivots) p.localRotation = Quaternion.Euler(0, steerVisual, 0);
        }
    }
}

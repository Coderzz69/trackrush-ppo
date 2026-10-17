using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Net.Sockets;
using Unity.MLAgents;
using Unity.MLAgents.Policies;
using UnityEngine;
using UnityEngine.InputSystem;

namespace TrackRush
{
    /// Lays out tracks and fills them with RacerAgents that share one behaviour ("TMRacer").
    /// Cars pass through each other, so one track can hold hundreds of learners.
    ///
    /// Two uses:
    ///  - the Training scene / headless build that Python launches (Spectator off), and
    ///  - the in-game "Watch AI Train" mode (Spectator on): waits until your trainer is listening,
    ///    then spawns the cars and shows a live view with stats.
    public class TrainingManager : MonoBehaviour
    {
        public const int EditorTrainerPort = 5004; // where ML-Agents looks for a trainer when Python didn't launch the game

        [Tooltip("Tracks to train on (indices into Levels.All). Each gets its own copy.")]
        public int[] Tracks = { 0, 1, 2, 3, 4 };
        public int AgentsPerTrack = 8;
        [Tooltip("Every episode starts on the start line (the classic '1000 cars launch' look). Off = random safe spots, which learns the whole track faster.")]
        public bool MassStart = false;
        [Tooltip("Plain box cars and cheaper collision checks, for running hundreds of agents.")]
        public bool LightweightCars = false;
        [Tooltip("Drive agent 0 yourself to check observations and rewards without Python.")]
        public bool KeyboardTest = false;
        [Tooltip("Don't create agents until a trainer is listening on port 5004 (ML-Agents only tries to connect once).")]
        public bool WaitForTrainer = false;
        [Tooltip("Spectator view: follow-the-leader camera, stats, speed controls, back to menu.")]
        public bool Spectator = false;
        public float TrackSpacing = 4000f;

        readonly List<RacerAgent> agents = new();
        readonly List<CarController> cars = new();
        readonly List<Track> tracks = new();
        Action onExit;
        ChaseCamera chase;
        int watched;
        bool followLeader = true, overview, freeCam, hideUI, spawned, wasConnected, exiting;
        float leaderTimer;
        Texture2D white;

        // spectator stats
        int episodes, finishes, lastAcademySteps;
        float bestTime, stepsPerSecond, statTimer;
        readonly Queue<float> recentRewards = new();
        readonly List<float> curve = new();
        float curveSum;
        int curveCount;
        const int CurveBucket = 100, CurvePoints = 90;

        static readonly float[] speeds = { 1f, 2f, 5f, 10f, 20f };

        // record mode (standalone game launched with -record <dir>): every rendered frame is saved as a JPG
        // at a fixed frame rate, and the HUD becomes a clean caption for videos (see Training/record.py)
        string recordDir, recordCaption = "";
        int recordFps = 30, recordFrame;

        /// Starts the in-game watch mode on one track.
        public static TrainingManager LaunchSpectator(int track, int cars, bool massStart, Action onExit)
        {
            var tm = new GameObject("Watch AI Train").AddComponent<TrainingManager>();
            tm.Tracks = new[] { track };
            tm.AgentsPerTrack = cars;
            tm.MassStart = massStart;
            tm.LightweightCars = cars > 250; // real car models up to 250; boxes beyond that keep the frame rate up
            tm.WaitForTrainer = true;
            tm.Spectator = true;
            tm.onExit = onExit;
            Time.maximumDeltaTime = 0.1f; // a slow frame slows the sim instead of piling up physics steps
            return tm;
        }

        void Awake()
        {
            Time.fixedDeltaTime = 1f / 60f;
            CarController.ConfigureCarLayer();
            var cam = Camera.main;
            if (cam == null)
            {
                cam = new GameObject("Main Camera", typeof(Camera), typeof(AudioListener)).GetComponent<Camera>();
                cam.tag = "MainCamera";
            }
            cam.nearClipPlane = 0.1f;
            cam.farClipPlane = 4000f;
            chase = cam.GetComponent<ChaseCamera>();
            if (chase == null) chase = cam.gameObject.AddComponent<ChaseCamera>();
        }

        void OnEnable() => RacerAgent.EpisodeEnded += OnEpisodeEnded;
        void OnDisable() => RacerAgent.EpisodeEnded -= OnEpisodeEnded;

        /// Player builds take "-agentsPerTrack N" and "-tracks 0,2,4" from the command line
        /// (Python passes them via UnityEnvironment's additional_args), so no rebuild is needed to change them.
        void ReadCommandLine()
        {
            var args = Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length - 1; i++)
            {
                if (args[i] == "-agentsPerTrack" && int.TryParse(args[i + 1], out int n) && n > 0)
                    AgentsPerTrack = n;
                else if (args[i] == "-tracks")
                {
                    var ids = new List<int>();
                    foreach (var s in args[i + 1].Split(','))
                        if (int.TryParse(s, out int t) && t >= 0 && t < Levels.All.Length) ids.Add(t);
                    if (ids.Count > 0) Tracks = ids.ToArray();
                }
            }
        }

        void ReadRecordArgs()
        {
            var args = Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length - 1; i++)
            {
                if (args[i] == "-record") recordDir = args[i + 1];
                else if (args[i] == "-caption") recordCaption = args[i + 1].Replace("|", "\n");
                else if (args[i] == "-recordFps" && int.TryParse(args[i + 1], out int f) && f > 0) recordFps = f;
            }
        }

        IEnumerator RecordFrames()
        {
            Directory.CreateDirectory(recordDir);
            var endOfFrame = new WaitForEndOfFrame();
            while (true)
            {
                yield return endOfFrame;
                if (!spawned) continue;
                Time.captureFramerate = recordFps; // game time advances exactly 1/fps per frame: smooth video at any speed
                var tex = ScreenCapture.CaptureScreenshotAsTexture();
                File.WriteAllBytes(Path.Combine(recordDir, $"frame_{recordFrame++:00000}.jpg"), tex.EncodeToJPG(90));
                Destroy(tex);
            }
        }

        IEnumerator Start()
        {
            if (!Application.isEditor && !Spectator) ReadCommandLine();
            if (Spectator && !Application.isEditor) ReadRecordArgs();
            if (recordDir != null) StartCoroutine(RecordFrames());
            for (int k = 0; k < Tracks.Length; k++)
                tracks.Add(Track.Build(Levels.All[Tracks[k]], new Vector3(k * TrackSpacing, 0, 0), scenery: Spectator));
            if (tracks.Count > 0) chase.SetOrbit(tracks[0].Center, tracks[0].Radius);

            if (WaitForTrainer && !KeyboardTest)
                while (!TrainerListening())
                    yield return new WaitForSecondsRealtime(0.5f);
            if (exiting) yield break;
            Spawn();
        }

        static bool TrainerListening()
        {
            try
            {
                using var client = new TcpClient();
                var connect = client.ConnectAsync("127.0.0.1", EditorTrainerPort);
                return connect.Wait(150) && client.Connected;
            }
            catch (Exception)
            {
                return false;
            }
        }

        void Spawn()
        {
            for (int k = 0; k < tracks.Count; k++)
                for (int a = 0; a < AgentsPerTrack; a++)
                {
                    int id = agents.Count;
                    var car = CarFactory.Create(Teams.AI[id % Teams.AI.Length], $"Agent {id}", $"A{id:000}", false, LightweightCars);
                    car.Track = tracks[k];
                    car.Frozen = false;
                    car.PlaceOnTrack(tracks[k].StartIndex);
                    var type = KeyboardTest && id == 0 ? BehaviorType.HeuristicOnly : BehaviorType.Default;
                    var agent = RacerAgent.Attach(car, true, type);
                    agent.MassStart = MassStart;
                    agents.Add(agent);
                    cars.Add(car);
                }
            spawned = true;
            if (agents.Count > 0 && (!Spectator || (!overview && !freeCam))) chase.Follow(cars[0]);
        }

        void OnEpisodeEnded(RacerAgent agent, RacerAgent.EpisodeStats s)
        {
            episodes++;
            recentRewards.Enqueue(s.Reward);
            while (recentRewards.Count > 200) recentRewards.Dequeue();
            if (s.Finished)
            {
                finishes++;
                if (s.FromStart && (bestTime <= 0 || s.FinishTime < bestTime)) bestTime = s.FinishTime;
            }
            curveSum += s.Reward;
            if (++curveCount >= CurveBucket)
            {
                curve.Add(curveSum / curveCount);
                if (curve.Count > CurvePoints) curve.RemoveAt(0);
                curveSum = 0;
                curveCount = 0;
            }
        }

        bool Connected => Academy.IsInitialized && Academy.Instance.IsCommunicatorOn;

        void Update()
        {
            var kb = Keyboard.current;
            if (kb == null) return;

            if (Spectator)
            {
                if (kb.escapeKey.wasPressedThisFrame) { Exit(); return; }
                if (kb.hKey.wasPressedThisFrame) hideUI = !hideUI;
                if (kb.oKey.wasPressedThisFrame) SetOverview(!overview);
                if (kb.fKey.wasPressedThisFrame) SetFreeCam(!freeCam);
                if (kb.lKey.wasPressedThisFrame) { followLeader = true; SetOverview(false); }
                if (kb.leftBracketKey.wasPressedThisFrame) StepSpeed(-1);
                if (kb.rightBracketKey.wasPressedThisFrame) StepSpeed(1);
                UpdateStats();
            }
            if (!spawned || agents.Count == 0) return;

            if (kb.tabKey.wasPressedThisFrame)
            {
                followLeader = false;
                watched = (watched + 1) % agents.Count;
                SetOverview(false);
            }
            if (kb.cKey.wasPressedThisFrame) chase.CycleMode();

            if (Spectator && followLeader && !overview && (leaderTimer -= Time.unscaledDeltaTime) <= 0)
            {
                leaderTimer = 0.5f;
                int best = 0;
                for (int i = 1; i < cars.Count; i++)
                    if (cars[i].Progress > cars[best].Progress) best = i;
                if (best != watched && cars[best].Progress > cars[watched].Progress + 15f) watched = best;
            }
            if (!overview && !freeCam && chase.Target != cars[watched]) chase.Follow(cars[watched]);
        }

        void SetOverview(bool on)
        {
            overview = on;
            freeCam = false;
            if (on && tracks.Count > 0) chase.SetOrbit(tracks[0].Center, tracks[0].Radius);
            else if (cars.Count > 0) chase.Follow(cars[watched]);
        }

        void SetFreeCam(bool on)
        {
            if (!on) { SetOverview(false); return; }
            freeCam = true;
            overview = false;
            followLeader = false;
            chase.SetFree();
        }

        void StepSpeed(int dir)
        {
            int i = 0;
            for (int k = 0; k < speeds.Length; k++)
                if (Mathf.Abs(speeds[k] - Time.timeScale) < Mathf.Abs(speeds[i] - Time.timeScale)) i = k;
            Time.timeScale = speeds[Mathf.Clamp(i + dir, 0, speeds.Length - 1)];
        }

        void UpdateStats()
        {
            if (Connected) wasConnected = true;
            statTimer += Time.unscaledDeltaTime;
            if (statTimer < 1f) return;
            int steps = Academy.IsInitialized ? Academy.Instance.StepCount : 0;
            stepsPerSecond = Mathf.Max(0, steps - lastAcademySteps) / statTimer;
            lastAcademySteps = steps;
            statTimer = 0;
        }

        public void Exit()
        {
            if (exiting) return;
            exiting = true;
            StartCoroutine(ExitRoutine());
        }

        IEnumerator ExitRoutine()
        {
            foreach (var c in cars) if (c != null) Destroy(c.gameObject);
            foreach (var t in tracks) if (t != null) Destroy(t.gameObject);
            cars.Clear();
            agents.Clear();
            yield return null; // let the agents unregister before the Academy goes
            if (Academy.IsInitialized) Academy.Instance.Dispose(); // so the next session reconnects
            Time.timeScale = 1f;
            Time.maximumDeltaTime = 1f / 3f;
            onExit?.Invoke();
            Destroy(gameObject);
        }

        // ---------------------------------------------------------------- GUI

        void OnGUI()
        {
            if (white == null) white = MatUtil.Solid(Color.white);
            if (!Spectator)
            {
                DrawTrainingOverlay();
                return;
            }
            float scale = Screen.height / 1080f;
            GUI.matrix = Matrix4x4.Scale(new Vector3(scale, scale, 1));
            float W = Screen.width / scale, H = 1080f;
            if (!spawned) DrawWaiting(W, H);
            else if (recordDir != null) DrawRecording(W);
            else if (!hideUI) DrawSpectator(W, H);
        }

        void Box(Rect r, Color c)
        {
            GUI.color = c;
            GUI.DrawTexture(r, white);
            GUI.color = Color.white;
        }

        static GUIStyle Text(int size, FontStyle fs = FontStyle.Normal, TextAnchor a = TextAnchor.UpperLeft) =>
            new(GUI.skin.label) { fontSize = size, fontStyle = fs, alignment = a, richText = true, wordWrap = true };

        static readonly Color PanelColor = new(0.05f, 0.06f, 0.09f, 0.88f);
        static readonly Color AccentColor = new(0.15f, 0.75f, 1f);

        bool Button(Rect r, string text)
        {
            bool hover = r.Contains(Event.current.mousePosition);
            Box(r, hover ? new Color(0.3f, 0.32f, 0.38f, 0.95f) : new Color(0.15f, 0.16f, 0.2f, 0.95f));
            GUI.Label(r, text, Text(20, FontStyle.Bold, TextAnchor.MiddleCenter));
            return GUI.Button(r, GUIContent.none, GUIStyle.none);
        }

        void DrawWaiting(float W, float H)
        {
            var p = new Rect(W / 2 - 440, H / 2 - 230, 880, 440);
            Box(p, PanelColor);
            Box(new Rect(p.x, p.y, p.width, 8), AccentColor);
            int dots = (int)(Time.unscaledTime * 2) % 4;
            GUI.Label(new Rect(p.x + 40, p.y + 30, p.width - 80, 50), "WAITING FOR YOUR TRAINER" + new string('.', dots), Text(36, FontStyle.BoldAndItalic));
            GUI.Label(new Rect(p.x + 40, p.y + 95, p.width - 80, 250),
                $"Start your Python PPO script and connect it to this game:\n\n" +
                $"<color=#7cf>    env = UnityEnvironment(file_name=None, base_port={EditorTrainerPort}, timeout_wait=600)</color>\n\n" +
                $"As soon as it is listening, {AgentsPerTrack} cars spawn on <b>{tracks[0].Level.Name}</b> and start learning.\n" +
                $"Behaviour <b>{RacerAgent.Behavior}</b>: {RacerAgent.ObservationSize} observations, {RacerAgent.ActionSize} actions. " +
                "Set time_scale to 1 in your EngineConfigurationChannel to watch in real time.",
                Text(20));
            if (Button(new Rect(p.x + 40, p.yMax - 80, 220, 52), "BACK  (Esc)")) Exit();
        }

        void DrawSpectator(float W, float H)
        {
            var track = tracks[0];
            var p = new Rect(20, 20, 460, 330);
            Box(p, PanelColor);
            Box(new Rect(p.x, p.y, 6, p.height), AccentColor);
            string status = Connected ? "<color=#5f5>TRAINER CONNECTED</color>"
                : wasConnected ? "<color=#f77>TRAINER DISCONNECTED</color>"
                : "<color=#fd5>CONNECTING...</color>";
            float avg = 0;
            foreach (var r in recentRewards) avg += r;
            if (recentRewards.Count > 0) avg /= recentRewards.Count;
            float realtime = stepsPerSecond / 60f;
            GUI.Label(new Rect(p.x + 22, p.y + 12, p.width - 40, 40), "WATCH AI TRAIN", Text(28, FontStyle.BoldAndItalic));
            GUI.Label(new Rect(p.x + 22, p.y + 52, p.width - 40, 220),
                $"{status}\n" +
                $"<b>{agents.Count}</b> cars on {track.Level.Name}{(MassStart ? "  ·  mass start" : "")}\n" +
                $"sim speed  <b>{realtime:0.0}x</b> real time   (time scale x{Time.timeScale:0})\n" +
                $"episodes  <b>{episodes:N0}</b>     finishes  <b>{finishes:N0}</b>\n" +
                $"best AI time  <b>{(bestTime > 0 ? FormatTime(bestTime) : "-")}</b>{(MassStart ? "" : "  <color=#999>(mass start only)</color>")}\n" +
                $"avg reward (last {recentRewards.Count})  <b>{avg:0.00}</b>",
                Text(19));

            // learning curve: average episode reward per bucket of episodes
            var g = new Rect(p.x + 22, p.y + 232, p.width - 44, 80);
            Box(g, new Color(1, 1, 1, 0.05f));
            GUI.Label(new Rect(g.x + 4, g.y - 2, g.width, 20), $"<color=#9ab>avg reward per {CurveBucket} episodes</color>", Text(13));
            if (curve.Count > 1)
            {
                float lo = float.MaxValue, hi = float.MinValue;
                foreach (var v in curve) { lo = Mathf.Min(lo, v); hi = Mathf.Max(hi, v); }
                if (hi - lo < 1e-3f) hi = lo + 1;
                float bw = g.width / CurvePoints;
                for (int i = 0; i < curve.Count; i++)
                {
                    float h = Mathf.Max(2f, (curve[i] - lo) / (hi - lo) * (g.height - 22));
                    Box(new Rect(g.x + i * bw, g.yMax - h, bw - 1, h), AccentColor);
                }
            }

            // watched car
            if (cars.Count > 0 && !overview && !freeCam)
            {
                var c = cars[watched];
                var w = new Rect(W - 340, 20, 320, 110);
                Box(w, PanelColor);
                GUI.Label(new Rect(w.x + 18, w.y + 10, w.width - 30, 100),
                    $"<b>{(followLeader ? "LEADER" : "WATCHING")}  {c.Abbrev}</b>\n" +
                    $"{c.SpeedKmh:0} km/h   ·   {c.Progress / track.Length * 100f:0}% of the track\n" +
                    $"reward this episode  {agents[watched].GetCumulativeReward():0.00}", Text(18));
            }

            var keys = new Rect(20, H - 50, W - 40, 30);
            GUI.Label(keys, "<color=#bbb>L follow leader   ·   Tab next car   ·   O overview   ·   F free cam (RMB look, WASD/QE, Shift)   ·   C camera   ·   [ ] speed   ·   H hide   ·   Esc menu</color>", Text(17));
            if (Button(new Rect(W - 180, H - 110, 160, 46), "MENU")) Exit();
        }

        void DrawRecording(float W)
        {
            var track = tracks[0];
            var p = new Rect(20, 20, 620, 150);
            Box(p, PanelColor);
            Box(new Rect(p.x, p.y, 6, p.height), AccentColor);
            GUI.Label(new Rect(p.x + 22, p.y + 12, p.width - 40, 40), $"TRACK RUSH  ·  {track.Level.Name}", Text(28, FontStyle.BoldAndItalic));
            GUI.Label(new Rect(p.x + 22, p.y + 54, p.width - 40, 90), recordCaption, Text(21));

            if (cars.Count > 0 && !overview && !freeCam)
            {
                var c = cars[watched];
                var w = new Rect(W - 380, 20, 360, 150);
                Box(w, PanelColor);
                GUI.Label(new Rect(w.x + 18, w.y + 12, w.width - 30, 130),
                    $"<b>BEST CAR  {c.Abbrev}</b>   <color=#9ab>of {agents.Count}</color>\n" +
                    $"{c.SpeedKmh:0} km/h   ·   {c.Progress / track.Length * 100f:0}% of the track\n" +
                    $"reward  {agents[watched].GetCumulativeReward():0.00}\n" +
                    $"best lap  <b>{(bestTime > 0 ? FormatTime(bestTime) : "-")}</b>", Text(21));
            }
        }

        static string FormatTime(float t) => $"{(int)(t / 60)}:{t % 60:00.000}";

        void DrawTrainingOverlay()
        {
            if (agents.Count == 0) return;
            var a = agents[watched];
            var car = a.GetComponent<CarController>();
            GUI.color = new Color(0, 0, 0, 0.75f);
            GUI.DrawTexture(new Rect(10, 10, 470, 128), white);
            GUI.color = Color.white;
            var style = new GUIStyle(GUI.skin.label) { fontSize = 15, richText = true };
            string trainer = Connected ? "<color=#5f5>trainer connected</color>"
                : KeyboardTest ? "<color=#fd5>keyboard test (agent 0)</color>"
                : "<color=#f77>no trainer - start Python first</color>";
            GUI.Label(new Rect(20, 16, 450, 120),
                $"<b>TRACK RUSH RL TRAINING</b>   {trainer}\n" +
                $"{agents.Count} agents on {Tracks.Length} track(s)   time x{Time.timeScale:0}\n" +
                $"watching <b>{car.Abbrev}</b> on {car.Track.Level.Name}\n" +
                $"step {a.StepCount}   reward {a.GetCumulativeReward():0.00}   {car.SpeedKmh:0} km/h\n" +
                "<color=#aaa>Tab: next agent   C: camera</color>", style);
        }
    }
}

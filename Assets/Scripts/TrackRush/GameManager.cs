using System.Collections.Generic;
using System.Globalization;
using Unity.InferenceEngine;
using Unity.MLAgents.Policies;
using UnityEngine;
using UnityEngine.InputSystem;

namespace TrackRush
{
    public enum GameState { Menu, Countdown, Running, Finished, Results, Watching }
    public enum MenuMode { Play, Watch }

    /// Time attack in the Trackmania mould: 3-2-1, checkpoints with splits against your best,
    /// instant restart, respawn at the last checkpoint, ghost of your best run and medals.
    /// Spawns itself in any scene that has no TrainingManager.
    public class GameManager : MonoBehaviour
    {
        public static GameManager Instance { get; private set; }

        public const string RivalModelPath = "ML/TMRacer"; // Assets/Resources/ML/TMRacer.onnx
        const string UnlockKey = "tr_unlocked", GhostKey = "tr_show_ghost", RivalKey = "tr_rival";
        const string WatchCarsKey = "tr_watch_cars", WatchMassKey = "tr_watch_mass";
        const float CountdownLength = 1.5f;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        static void ResetStatics() => Instance = null;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Bootstrap()
        {
            if (Instance == null && FindAnyObjectByType<TrainingManager>() == null)
                new GameObject("Track Rush").AddComponent<GameManager>();
        }

        public GameState State { get; private set; }
        public Track CurrentTrack { get; private set; }
        public int LevelIndex { get; private set; }
        LevelData Level => Levels.All[LevelIndex];

        ChaseCamera chase;
        CarController player, rival;
        ModelAsset rivalModel;
        bool useRival, showGhost = true, paused, newPB, newUnlock;
        int unlocked;
        MenuMode menuMode = MenuMode.Play;
        int watchCars = 1000;
        bool watchMassStart = true;

        float runTime, countdown, finishedFor, offTrackFor;
        int nextCheckpoint, respawnIndex, physicsTick;
        readonly List<float> splits = new();
        float[] pbSplits = new float[0];
        float pbTime, finishTime, previousPB;
        string splitText;
        Color splitColor;
        float splitShownFor;

        GhostRun pbGhost, recording;
        Transform ghostCar;

        // ---------------------------------------------------------------- lifecycle

        void Awake()
        {
            Instance = this;
            CarController.ConfigureCarLayer();
            Time.fixedDeltaTime = 1f / 60f;
            unlocked = Mathf.Clamp(PlayerPrefs.GetInt(UnlockKey, 1), 1, Levels.All.Length);
            showGhost = PlayerPrefs.GetInt(GhostKey, 1) == 1;
            rivalModel = Resources.Load<ModelAsset>(RivalModelPath);
            useRival = rivalModel != null && PlayerPrefs.GetInt(RivalKey, 0) == 1;
            watchCars = PlayerPrefs.GetInt(WatchCarsKey, 1000);
            watchMassStart = PlayerPrefs.GetInt(WatchMassKey, 1) == 1;

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

            if (FindAnyObjectByType<Light>() == null)
            {
                var sun = new GameObject("Sun").AddComponent<Light>();
                sun.type = LightType.Directional;
                sun.shadows = LightShadows.Soft;
                sun.transform.rotation = Quaternion.Euler(50f, -30f, 0);
            }
        }

        /// ML-Agents only connects a standalone player to a trainer when it was launched with --mlagents-port
        /// (TrackRush.sh and Python's TrackRushEnv(watch=...) pass it); the Editor always can.
        static bool CanReachTrainer => Application.isEditor || System.Array.IndexOf(System.Environment.GetCommandLineArgs(), "--mlagents-port") >= 0;

        /// Launch options for the standalone game: "-track 2" opens that track, and "-watch 100" goes
        /// straight into Watch AI Train with that many cars ("-massStart 0" for random start spots).
        void Start()
        {
            int track = 0, watch = 0;
            var args = System.Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length - 1; i++)
            {
                if (args[i] == "-track" && int.TryParse(args[i + 1], out int t)) track = Mathf.Clamp(t, 0, Levels.All.Length - 1);
                else if (args[i] == "-watch" && int.TryParse(args[i + 1], out int n) && n > 0) watch = n;
                else if (args[i] == "-massStart") watchMassStart = args[i + 1] != "0";
            }
            LoadLevel(track);
            if (watch > 0)
            {
                watchCars = watch;
                StartWatching();
            }
        }

        void LoadLevel(int index)
        {
            ClearRun();
            if (CurrentTrack != null) Destroy(CurrentTrack.gameObject);
            LevelIndex = index;
            CurrentTrack = Track.Build(Level);
            chase.SetOrbit(CurrentTrack.Center, CurrentTrack.Radius);
            State = GameState.Menu;
            SetPaused(false);
        }

        void ClearRun()
        {
            if (player != null) Destroy(player.gameObject);
            if (rival != null) Destroy(rival.gameObject);
            if (ghostCar != null) Destroy(ghostCar.gameObject);
            player = rival = null;
            ghostCar = null;
        }

        string PBKey => $"tr_pb_{LevelIndex}";
        string SplitsKey => $"tr_splits_{LevelIndex}";

        void StartRun()
        {
            ClearRun();
            SetPaused(false);
            newPB = newUnlock = false;

            player = CarFactory.Create(Teams.Player, "You", "YOU", true);
            player.Track = CurrentTrack;
            player.Frozen = true;
            player.PlaceOnTrack(CurrentTrack.StartIndex);

            if (useRival && rivalModel != null)
            {
                rival = CarFactory.Create(Teams.AI[0], "AI Rival", "AI", false);
                rival.Track = CurrentTrack;
                rival.Frozen = true;
                RacerAgent.Attach(rival, false, BehaviorType.InferenceOnly, rivalModel);
                rival.PlaceOnTrack(CurrentTrack.StartIndex);
            }

            pbTime = PlayerPrefs.GetFloat(PBKey, 0);
            pbSplits = ParseSplits(PlayerPrefs.GetString(SplitsKey, ""));
            pbGhost = GhostRun.Load(LevelIndex);
            if (pbGhost != null && showGhost) ghostCar = CarFactory.CreateGhost(new Color(0.6f, 0.8f, 1f, 0.35f));

            recording = new GhostRun();
            splits.Clear();
            nextCheckpoint = 0;
            respawnIndex = CurrentTrack.StartIndex;
            runTime = 0;
            countdown = 0;
            physicsTick = 0;
            offTrackFor = 0;
            splitShownFor = 0;
            chase.Follow(player);
            State = GameState.Countdown;
        }

        void Respawn()
        {
            if (player == null || State != GameState.Running) return;
            player.PlaceOnTrack(respawnIndex);
            offTrackFor = 0;
        }

        void SetPaused(bool p)
        {
            paused = p;
            Time.timeScale = p ? 0f : 1f;
        }

        // ---------------------------------------------------------------- update

        void Update()
        {
            if (State == GameState.Watching) return;
            var kb = Keyboard.current;
            var gp = Gamepad.current;
            bool Pressed(Key k) => kb != null && kb[k].wasPressedThisFrame;

            bool inRun = State is GameState.Countdown or GameState.Running or GameState.Finished;
            if (inRun && (Pressed(Key.Escape) || (gp != null && gp.startButton.wasPressedThisFrame))) SetPaused(!paused);
            if (paused) return;

            if (inRun || State == GameState.Results)
            {
                if (Pressed(Key.R) || Pressed(Key.Delete) || (gp != null && gp.buttonNorth.wasPressedThisFrame)) StartRun();
                if (Pressed(Key.G)) ToggleGhost();
            }
            if (State == GameState.Results && (Pressed(Key.Enter) || Pressed(Key.NumpadEnter)) && NextAvailable) NextLevel();
            if (player == null) return;

            if (Pressed(Key.C) || (gp != null && gp.buttonWest.wasPressedThisFrame)) chase.CycleMode();
            if (Pressed(Key.Backspace) || Pressed(Key.Enter) || (gp != null && gp.buttonEast.wasPressedThisFrame)) Respawn();

            if (State == GameState.Countdown)
            {
                countdown += Time.deltaTime;
                if (countdown >= CountdownLength)
                {
                    State = GameState.Running;
                    player.Frozen = false;
                    if (rival != null) rival.Frozen = false;
                }
            }

            if (State == GameState.Running) ReadInput(kb, gp);
            else if (State == GameState.Finished)
            {
                player.Controls = new CarInput { Brake = 0.6f };
                finishedFor += Time.deltaTime;
                if (finishedFor > 1.2f) State = GameState.Results;
            }

            bool lost = !player.OnRoad && (player.OnGrass || player.HeightAboveTrack < -3f);
            offTrackFor = lost && State == GameState.Running ? offTrackFor + Time.deltaTime : 0;
            splitShownFor -= Time.deltaTime;
        }

        void ReadInput(Keyboard kb, Gamepad gp)
        {
            float thr = 0, brk = 0, steer = 0;
            if (kb != null)
            {
                if (kb.wKey.isPressed || kb.upArrowKey.isPressed) thr = 1;
                if (kb.sKey.isPressed || kb.downArrowKey.isPressed || kb.spaceKey.isPressed) brk = 1;
                if (kb.aKey.isPressed || kb.leftArrowKey.isPressed) steer -= 1;
                if (kb.dKey.isPressed || kb.rightArrowKey.isPressed) steer += 1;
            }
            if (gp != null)
            {
                thr = Mathf.Max(thr, gp.rightTrigger.ReadValue());
                brk = Mathf.Max(brk, gp.leftTrigger.ReadValue());
                float sx = gp.leftStick.x.ReadValue();
                if (Mathf.Abs(sx) > 0.1f) steer = sx;
            }
            player.Controls = new CarInput { Throttle = thr, Brake = brk, Steer = steer };
        }

        void FixedUpdate()
        {
            if (player == null || paused) return;
            var t = CurrentTrack;
            int prev = player.TrackIndex;
            t.Locate(player);
            if (rival != null) t.Locate(rival);

            if (State != GameState.Running) return;
            runTime += Time.fixedDeltaTime;
            if (physicsTick++ % 2 == 0) recording.Record(player.transform);

            int cur = player.TrackIndex;
            bool continuous = cur - prev < 60 && t.InGate(player); // ignore tracking jumps (respawn) and shortcuts
            if (nextCheckpoint < t.Checkpoints.Length)
            {
                int cp = t.Checkpoints[nextCheckpoint];
                if (continuous && prev < cp && cur >= cp) PassCheckpoint();
            }
            else if (continuous && prev < t.FinishIndex && cur >= t.FinishIndex) Finish();

            if (ghostCar != null && pbGhost != null && pbGhost.Sample(runTime, out var gp, out var gr))
                ghostCar.SetPositionAndRotation(gp, gr);
        }

        void PassCheckpoint()
        {
            splits.Add(runTime);
            respawnIndex = CurrentTrack.Checkpoints[nextCheckpoint];
            ShowSplit(nextCheckpoint, runTime);
            nextCheckpoint++;
        }

        void ShowSplit(int index, float time)
        {
            if (index < pbSplits.Length && pbSplits[index] > 0)
            {
                float d = time - pbSplits[index];
                splitText = (d <= 0 ? "-" : "+") + FormatTime(Mathf.Abs(d), true);
                splitColor = d <= 0 ? new Color(0.35f, 0.6f, 1f) : new Color(1f, 0.3f, 0.25f);
            }
            else
            {
                splitText = FormatTime(time);
                splitColor = Color.white;
            }
            splitShownFor = 3f;
        }

        void Finish()
        {
            finishTime = runTime;
            splits.Add(runTime);
            ShowSplit(splits.Count - 1, runTime);
            State = GameState.Finished;
            finishedFor = 0;

            previousPB = pbTime;
            if (pbTime <= 0 || finishTime < pbTime)
            {
                newPB = true;
                pbTime = finishTime;
                PlayerPrefs.SetFloat(PBKey, finishTime);
                PlayerPrefs.SetString(SplitsKey, string.Join(",", splits.ConvertAll(s => s.ToString("R", CultureInfo.InvariantCulture))));
                recording.Save(LevelIndex);
            }
            if (Level.MedalFor(finishTime) >= Medal.Bronze && LevelIndex + 1 < Levels.All.Length && unlocked < LevelIndex + 2)
            {
                unlocked = LevelIndex + 2;
                PlayerPrefs.SetInt(UnlockKey, unlocked);
                newUnlock = true;
            }
            PlayerPrefs.Save();
        }

        bool NextAvailable => LevelIndex + 1 < Levels.All.Length && LevelIndex + 1 < unlocked;

        void NextLevel()
        {
            LoadLevel(LevelIndex + 1);
            StartRun();
        }

        void ToggleGhost()
        {
            showGhost = !showGhost;
            PlayerPrefs.SetInt(GhostKey, showGhost ? 1 : 0);
            if (!showGhost && ghostCar != null) Destroy(ghostCar.gameObject);
            if (showGhost && ghostCar == null && pbGhost != null && player != null)
                ghostCar = CarFactory.CreateGhost(new Color(0.6f, 0.8f, 1f, 0.35f));
        }

        static float[] ParseSplits(string s)
        {
            if (string.IsNullOrEmpty(s)) return new float[0];
            var parts = s.Split(',');
            var r = new float[parts.Length];
            for (int i = 0; i < parts.Length; i++)
                float.TryParse(parts[i], NumberStyles.Float, CultureInfo.InvariantCulture, out r[i]);
            return r;
        }

        static string FormatTime(float t, bool compact = false)
        {
            if (t <= 0 || t > 1e6f) return "-:--.---";
            int m = (int)(t / 60);
            if (compact && m == 0) return (t % 60).ToString("0.000", CultureInfo.InvariantCulture);
            return $"{m}:{(t % 60).ToString("00.000", CultureInfo.InvariantCulture)}";
        }

        // ---------------------------------------------------------------- GUI

        Texture2D white, circle;
        GUIStyle sTitle, sH1, sH2, sBody, sSmall, sTimer, sHuge;
        float W, H;

        static readonly Color Panel = new(0.05f, 0.06f, 0.09f, 0.88f);
        static readonly Color Accent = new(0.15f, 0.75f, 1f);
        static readonly Color Grey = new(0.65f, 0.67f, 0.72f);

        static Color MedalColor(Medal m) => m switch
        {
            Medal.Author => new Color(0.15f, 0.8f, 0.45f),
            Medal.Gold => new Color(1f, 0.8f, 0.2f),
            Medal.Silver => new Color(0.78f, 0.8f, 0.85f),
            Medal.Bronze => new Color(0.8f, 0.5f, 0.25f),
            _ => new Color(1, 1, 1, 0.15f),
        };

        GUIStyle Style(int size, FontStyle fs, TextAnchor anchor, Color col)
        {
            var s = new GUIStyle(GUI.skin.label)
            {
                fontSize = size, fontStyle = fs, alignment = anchor, wordWrap = false,
                clipping = TextClipping.Overflow, richText = true,
            };
            s.normal.textColor = col;
            return s;
        }

        void InitGUI()
        {
            if (white != null) return;
            white = MatUtil.Solid(Color.white);
            circle = MatUtil.Circle(64);
            sTitle = Style(80, FontStyle.BoldAndItalic, TextAnchor.UpperLeft, Color.white);
            sH1 = Style(40, FontStyle.BoldAndItalic, TextAnchor.MiddleCenter, Color.white);
            sH2 = Style(28, FontStyle.Bold, TextAnchor.UpperLeft, Color.white);
            sBody = Style(22, FontStyle.Normal, TextAnchor.UpperLeft, Color.white);
            sSmall = Style(18, FontStyle.Normal, TextAnchor.UpperLeft, Grey);
            sTimer = Style(64, FontStyle.Bold, TextAnchor.MiddleCenter, Color.white);
            sHuge = Style(150, FontStyle.BoldAndItalic, TextAnchor.MiddleCenter, Color.white);
        }

        void Box(Rect r, Color c)
        {
            GUI.color = c;
            GUI.DrawTexture(r, white);
            GUI.color = Color.white;
        }

        void Dot(Rect r, Color c)
        {
            GUI.color = c;
            GUI.DrawTexture(r, circle);
            GUI.color = Color.white;
        }

        void Label(Rect r, string text, GUIStyle s, Color? col = null)
        {
            var old = s.normal.textColor;
            if (col.HasValue) s.normal.textColor = col.Value;
            GUI.Label(r, text, s);
            s.normal.textColor = old;
        }

        void Shadowed(Rect r, string text, GUIStyle s, Color? col = null)
        {
            Label(new Rect(r.x + 3, r.y + 3, r.width, r.height), text, s, new Color(0, 0, 0, 0.6f));
            Label(r, text, s, col);
        }

        bool Button(Rect r, string text, bool primary = false, bool enabled = true)
        {
            bool hover = enabled && r.Contains(Event.current.mousePosition);
            Color bg = !enabled ? new Color(0.2f, 0.2f, 0.22f, 0.8f)
                : primary ? (hover ? new Color(0.35f, 0.85f, 1f) : Accent)
                : (hover ? new Color(0.3f, 0.32f, 0.38f, 0.95f) : new Color(0.15f, 0.16f, 0.2f, 0.95f));
            Box(r, bg);
            Label(r, text, new GUIStyle(sH2) { alignment = TextAnchor.MiddleCenter, fontSize = 24 },
                !enabled ? Grey : primary ? new Color(0.02f, 0.05f, 0.1f) : Color.white);
            return GUI.Button(r, GUIContent.none, GUIStyle.none) && enabled;
        }

        void OnGUI()
        {
            InitGUI();
            float scale = Screen.height / 1080f;
            GUI.matrix = Matrix4x4.Scale(new Vector3(scale, scale, 1));
            W = Screen.width / scale;
            H = 1080f;
            if (State == GameState.Watching) return;

            switch (State)
            {
                case GameState.Menu: DrawMenu(); break;
                case GameState.Results: DrawHUD(); DrawResults(); break;
                default: DrawHUD(); break;
            }
            if (paused) DrawPause();
        }

        void DrawMedalRow(Rect r, Medal m, float time, bool achieved)
        {
            Dot(new Rect(r.x, r.y + 4, 22, 22), MedalColor(m));
            Label(new Rect(r.x + 34, r.y, 120, 30), m.ToString().ToUpper(), sSmall, achieved ? Color.white : Grey);
            Label(new Rect(r.x, r.y, r.width, 30), FormatTime(time), new GUIStyle(sBody) { alignment = TextAnchor.UpperRight, fontSize = 20 },
                achieved ? Color.white : Grey);
        }

        void DrawMenu()
        {
            Box(new Rect(0, 0, 660, H), Panel);
            Label(new Rect(40, 26, 600, 90), "TRACK RUSH", sTitle);
            Box(new Rect(44, 122, 150, 8), Accent);

            bool watch = menuMode == MenuMode.Watch;
            if (Button(new Rect(40, 146, 285, 56), "PLAY", !watch)) menuMode = MenuMode.Play;
            if (Button(new Rect(335, 146, 285, 56), "WATCH AI TRAIN", watch)) menuMode = MenuMode.Watch;

            for (int i = 0; i < Levels.All.Length; i++)
            {
                var lvl = Levels.All[i];
                bool locked = !watch && i >= unlocked, sel = i == LevelIndex;
                var r = new Rect(40, 218 + i * 104, 580, 94);
                Box(r, sel ? new Color(0.17f, 0.2f, 0.27f, 0.95f) : new Color(0.1f, 0.11f, 0.15f, 0.9f));
                if (sel) Box(new Rect(r.x, r.y, 6, r.height), Accent);
                Label(new Rect(r.x + 24, r.y + 10, 440, 40), lvl.Name.ToUpper(), sH2, locked ? Grey : Color.white);
                if (locked)
                    Label(new Rect(r.x + 24, r.y + 52, 540, 30), "LOCKED  -  earn a medal on the previous track", sSmall);
                else
                    Label(new Rect(r.x + 24, r.y + 46, 480, 30), lvl.Subtitle, sSmall);
                if (!watch)
                {
                    float pb = PlayerPrefs.GetFloat($"tr_pb_{i}", 0);
                    if (!locked)
                        Label(new Rect(r.x + 24, r.y + 68, 540, 30), pb > 0 ? $"PERSONAL BEST  {FormatTime(pb)}" : "NOT FINISHED YET",
                            new GUIStyle(sSmall) { fontSize = 16 }, pb > 0 ? Color.white : Grey);
                    Dot(new Rect(r.xMax - 60, r.y + 25, 44, 44), MedalColor(lvl.MedalFor(pb)));
                }
                if (!locked && !sel && GUI.Button(r, GUIContent.none, GUIStyle.none)) LoadLevel(i);
            }

            var right = new GUIStyle(sTitle) { alignment = TextAnchor.UpperRight, fontSize = 58 };
            Shadowed(new Rect(W - 900, 40, 860, 80), Level.Name.ToUpper(), right);
            Shadowed(new Rect(W - 900, 112, 860, 40), $"{CurrentTrack.Length / 1000f:0.00} KM  ·  {CurrentTrack.Checkpoints.Length} CHECKPOINTS",
                new GUIStyle(sBody) { alignment = TextAnchor.UpperRight });

            if (watch) DrawWatchOptions();
            else DrawPlayOptions();
        }

        void DrawPlayOptions()
        {
            if (Button(new Rect(40, 748, 285, 52), showGhost ? "GHOST: ON" : "GHOST: OFF")) ToggleGhost();
            if (rivalModel != null && Button(new Rect(335, 748, 285, 52), useRival ? "AI RIVAL: ON" : "AI RIVAL: OFF"))
            {
                useRival = !useRival;
                PlayerPrefs.SetInt(RivalKey, useRival ? 1 : 0);
            }
            if (Button(new Rect(40, 812, 580, 76), "PLAY", true)) StartRun();
            Label(new Rect(44, 905, 600, 30), "W/S  accelerate / brake      A/D  steer", sSmall);
            Label(new Rect(44, 930, 600, 30), "R  restart      Backspace  respawn at checkpoint", sSmall);
            Label(new Rect(44, 955, 600, 30), "C  camera      G  ghost      Esc  pause      (gamepad supported)", sSmall);
            if (unlocked > 1 || PlayerPrefs.HasKey("tr_pb_0"))
                if (Button(new Rect(W - 250, H - 70, 220, 44), "Reset progress"))
                {
                    foreach (var k in new[] { UnlockKey }) PlayerPrefs.DeleteKey(k);
                    for (int i = 0; i < Levels.All.Length; i++) { PlayerPrefs.DeleteKey($"tr_pb_{i}"); PlayerPrefs.DeleteKey($"tr_splits_{i}"); }
                    GhostRun.DeleteAll();
                    unlocked = 1;
                    LoadLevel(0);
                }

            var mp = new Rect(W - 330, 170, 290, 172);
            Box(mp, Panel);
            float best = PlayerPrefs.GetFloat($"tr_pb_{LevelIndex}", 0);
            int row = 0;
            foreach (var m in new[] { Medal.Author, Medal.Gold, Medal.Silver, Medal.Bronze })
                DrawMedalRow(new Rect(mp.x + 18, mp.y + 14 + row++ * 38, mp.width - 36, 34), m, Level.TimeFor(m), best > 0 && best <= Level.TimeFor(m));
        }

        static readonly int[] watchCarOptions = { 100, 250, 500, 1000 };

        void DrawWatchOptions()
        {
            Label(new Rect(44, 738, 560, 30), "CARS", sSmall);
            for (int k = 0; k < watchCarOptions.Length; k++)
            {
                int n = watchCarOptions[k];
                if (Button(new Rect(40 + k * 147.5f, 764, 137.5f, 50), n.ToString(), watchCars == n))
                {
                    watchCars = n;
                    PlayerPrefs.SetInt(WatchCarsKey, n);
                }
            }
            if (Button(new Rect(40, 826, 580, 50), watchMassStart ? "MASS START: ON  (all from the start line)" : "MASS START: OFF  (random spots on the track)"))
            {
                watchMassStart = !watchMassStart;
                PlayerPrefs.SetInt(WatchMassKey, watchMassStart ? 1 : 0);
            }
            if (Button(new Rect(40, 890, 580, 76), "START WATCHING", true)) StartWatching();
            if (CanReachTrainer)
                Label(new Rect(44, 980, 600, 30), "Needs your Python trainer (port 5004). The game waits for it.", sSmall);
            else
                Label(new Rect(44, 980, 600, 30), "<color=#f66>Start the game with TrackRush.sh so it can reach the trainer.</color>", sSmall);

            var ip = new Rect(W - 560, 170, 520, 290);
            Box(ip, Panel);
            Label(new Rect(ip.x + 24, ip.y + 18, ip.width - 48, 40), "HOW IT WORKS", sH2);
            Label(new Rect(ip.x + 24, ip.y + 62, ip.width - 48, 220),
                $"{watchCars} cars drive this track at once, all sharing your one\n" +
                "PPO policy. They pass through each other like ghosts.\n\n" +
                "1. Start your Python training script\n" +
                "    UnityEnvironment(file_name=None, base_port=5004)\n" +
                "2. Press START WATCHING\n" +
                "3. Watch them learn: follow the leader, see the stats\n" +
                "    and the learning curve. Esc returns here.",
                new GUIStyle(sBody) { fontSize = 19, wordWrap = true });
        }

        void StartWatching()
        {
            ClearRun();
            if (CurrentTrack != null) Destroy(CurrentTrack.gameObject);
            CurrentTrack = null;
            State = GameState.Watching;
            TrainingManager.LaunchSpectator(LevelIndex, watchCars, watchMassStart, () =>
            {
                menuMode = MenuMode.Watch;
                LoadLevel(LevelIndex);
            });
        }

        void DrawHUD()
        {
            if (player == null) return;
            var t = CurrentTrack;

            // timer
            var tr = new Rect(W / 2 - 180, H - 140, 360, 96);
            Box(tr, Panel);
            Label(tr, FormatTime(State == GameState.Countdown ? 0.0001f : State is GameState.Finished or GameState.Results ? finishTime : runTime), sTimer);

            // checkpoint split
            if (splitShownFor > 0 && !string.IsNullOrEmpty(splitText))
            {
                var sr = new Rect(W / 2 - 140, H - 210, 280, 56);
                Box(sr, Panel);
                Label(sr, splitText, new GUIStyle(sH1) { fontSize = 36, fontStyle = FontStyle.Bold }, splitColor);
            }

            // checkpoints
            var cr = new Rect(W / 2 - 150, 24, 300, 46);
            Box(cr, Panel);
            Label(cr, $"CHECKPOINT  {Mathf.Min(nextCheckpoint, t.Checkpoints.Length)} / {t.Checkpoints.Length}",
                new GUIStyle(sBody) { alignment = TextAnchor.MiddleCenter, fontStyle = FontStyle.Bold });

            // speed and boost
            var sp = new Rect(W - 300, H - 170, 276, 146);
            Box(sp, Panel);
            Label(new Rect(sp.x, sp.y + 4, sp.width - 24, 100), Mathf.RoundToInt(player.SpeedKmh).ToString(),
                new GUIStyle(sTimer) { fontSize = 84, alignment = TextAnchor.MiddleRight, fontStyle = FontStyle.BoldAndItalic });
            Label(new Rect(sp.x + 20, sp.y + 100, 100, 30), "KM/H", sSmall);
            if (player.BoostTime > 0)
            {
                Box(new Rect(sp.x + 20, sp.y + 128, sp.width - 40, 6), new Color(1, 1, 1, 0.1f));
                Box(new Rect(sp.x + 20, sp.y + 128, (sp.width - 40) * player.BoostTime / player.BoostDuration, 6), new Color(1f, 0.55f, 0.1f));
                Label(new Rect(sp.x, sp.y + 100, sp.width - 24, 30), "TURBO", new GUIStyle(sSmall) { alignment = TextAnchor.UpperRight }, new Color(1f, 0.6f, 0.15f));
            }

            // medals + personal best
            var mp = new Rect(W - 300, 24, 276, 212);
            Box(mp, Panel);
            int row = 0;
            foreach (var m in new[] { Medal.Author, Medal.Gold, Medal.Silver, Medal.Bronze })
                DrawMedalRow(new Rect(mp.x + 16, mp.y + 12 + row++ * 36, mp.width - 32, 32), m, Level.TimeFor(m), pbTime > 0 && pbTime <= Level.TimeFor(m));
            Box(new Rect(mp.x + 16, mp.y + 160, mp.width - 32, 1), new Color(1, 1, 1, 0.15f));
            Label(new Rect(mp.x + 16, mp.y + 170, 120, 30), "BEST", sSmall);
            Label(new Rect(mp.x + 16, mp.y + 168, mp.width - 32, 30), FormatTime(pbTime), new GUIStyle(sBody) { alignment = TextAnchor.UpperRight, fontStyle = FontStyle.Bold });

            Label(new Rect(24, H - 40, 900, 30), $"R restart   ·   Backspace respawn   ·   C camera   ·   G ghost {(showGhost ? "on" : "off")}   ·   Esc pause",
                new GUIStyle(sSmall) { fontSize = 16 });

            if (State == GameState.Countdown)
            {
                int n = 3 - Mathf.FloorToInt(countdown / (CountdownLength / 3f));
                Shadowed(new Rect(0, H * 0.3f, W, 200), Mathf.Clamp(n, 1, 3).ToString(), sHuge);
            }
            else if (State == GameState.Running && runTime < 0.8f)
                Shadowed(new Rect(0, H * 0.3f, W, 200), "GO!", sHuge, new Color(0.4f, 1f, 0.5f));

            if (offTrackFor > 1f)
                Shadowed(new Rect(0, H * 0.58f, W, 60), "Press BACKSPACE to respawn", new GUIStyle(sH1) { fontSize = 34 }, new Color(1f, 0.85f, 0.3f));
        }

        void DrawResults()
        {
            var medal = Level.MedalFor(finishTime);
            var p = new Rect(W / 2 - 360, H / 2 - 300, 720, 460);
            Box(p, Panel);
            Box(new Rect(p.x, p.y, p.width, 8), medal == Medal.None ? Accent : MedalColor(medal));
            Label(new Rect(p.x, p.y + 28, p.width, 40), Level.Name.ToUpper(), new GUIStyle(sSmall) { alignment = TextAnchor.UpperCenter, fontSize = 22 });
            Label(new Rect(p.x, p.y + 60, p.width, 110), FormatTime(finishTime), new GUIStyle(sTimer) { fontSize = 96 });

            string pbLine = newPB
                ? previousPB > 0 ? $"NEW PERSONAL BEST   -{FormatTime(previousPB - finishTime, true)}" : "FIRST FINISH!"
                : $"PERSONAL BEST {FormatTime(pbTime)}   (+{FormatTime(finishTime - pbTime, true)})";
            Label(new Rect(p.x, p.y + 172, p.width, 36), pbLine, new GUIStyle(sBody) { alignment = TextAnchor.UpperCenter, fontStyle = FontStyle.Bold },
                newPB ? new Color(0.35f, 0.65f, 1f) : Grey);

            Dot(new Rect(p.x + p.width / 2 - 150, p.y + 222, 52, 52), MedalColor(medal));
            Label(new Rect(p.x + p.width / 2 - 84, p.y + 222, 300, 52),
                medal == Medal.None ? "NO MEDAL" : $"{medal.ToString().ToUpper()} MEDAL", new GUIStyle(sH2) { alignment = TextAnchor.MiddleLeft, fontSize = 34 });
            if (medal != Medal.Author)
            {
                var next = medal + 1;
                Label(new Rect(p.x, p.y + 284, p.width, 30), $"{next.ToString().ToUpper()}: {FormatTime(Level.TimeFor(next))}",
                    new GUIStyle(sSmall) { alignment = TextAnchor.UpperCenter }, MedalColor(next));
            }
            if (newUnlock)
                Label(new Rect(p.x, p.y + 312, p.width, 30), "NEXT TRACK UNLOCKED!", new GUIStyle(sBody) { alignment = TextAnchor.UpperCenter }, new Color(0.4f, 1f, 0.5f));

            float by = p.yMax - 90;
            if (Button(new Rect(p.x + 30, by, 210, 60), "RETRY  (R)")) StartRun();
            if (Button(new Rect(p.x + 255, by, 210, 60), "MENU")) LoadLevel(LevelIndex);
            if (Button(new Rect(p.x + 480, by, 210, 60), "NEXT  (Enter)", true, NextAvailable)) NextLevel();
        }

        void DrawPause()
        {
            Box(new Rect(0, 0, W, H), new Color(0, 0, 0, 0.55f));
            var p = new Rect(W / 2 - 220, H / 2 - 200, 440, 380);
            Box(p, Panel);
            Box(new Rect(p.x, p.y, p.width, 8), Accent);
            Label(new Rect(p.x, p.y + 30, p.width, 50), "PAUSED", sH1);
            if (Button(new Rect(p.x + 40, p.y + 110, 360, 64), "RESUME", true)) SetPaused(false);
            if (Button(new Rect(p.x + 40, p.y + 190, 360, 64), "RESTART")) StartRun();
            if (Button(new Rect(p.x + 40, p.y + 270, 360, 64), "QUIT TO MENU")) LoadLevel(LevelIndex);
        }
    }
}

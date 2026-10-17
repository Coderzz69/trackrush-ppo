using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace TrackRush.EditorTools
{
    /// Builds the Training scene as a standalone Linux player that Python can launch headless
    /// (faster than training against the Editor, and several copies can run at once).
    public static class TrainingBuild
    {
        public const string Scene = "Assets/Scenes/Training.unity";
        public const string Output = "Training/Build/TrackRushTrain.x86_64";

        [MenuItem("Track Rush/Build Training Environment (Linux)")]
        public static void Build()
        {
            var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
            {
                scenes = new[] { Scene },
                locationPathName = Output,
                target = BuildTarget.StandaloneLinux64,
                options = BuildOptions.None,
            });
            if (report.summary.result == BuildResult.Succeeded)
                Debug.Log($"Training environment built: {Output} ({report.summary.totalSize / (1024 * 1024)} MB)");
            else
                Debug.LogError($"Training build failed: {report.summary.result}");
        }
    }

    /// Builds the whole game (menu, time attack, Watch AI Train) as a standalone Linux player, so it can
    /// run without the Editor. Watch AI Train works the same: it connects to a Python trainer on port 5004.
    public static class GameBuild
    {
        public const string Scene = "Assets/Scenes/SampleScene.unity";
        public const string Output = "Builds/TrackRush/TrackRush.x86_64";
        public const string Launcher = "Builds/TrackRush/TrackRush.sh";

        /// A standalone ML-Agents player only talks to a trainer when launched with --mlagents-port, so this
        /// script adds it; without a trainer listening the game just plays normally.
        static void WriteLauncher()
        {
            System.IO.File.WriteAllText(Launcher,
                "#!/bin/sh\n" +
                "# Starts Track Rush so that Watch AI Train can connect to a Python trainer on port 5004.\n" +
                "# Extra options: -watch 100 (straight into Watch AI Train), -track 2, -massStart 0, -screen-fullscreen 0\n" +
                "cd \"$(dirname \"$0\")\" && exec ./TrackRush.x86_64 --mlagents-port 5004 \"$@\"\n");
            System.Diagnostics.Process.Start("chmod", $"+x \"{System.IO.Path.GetFullPath(Launcher)}\"")?.WaitForExit();
        }

        [MenuItem("Track Rush/Build Game (Linux)")]
        public static void Build()
        {
            var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
            {
                scenes = new[] { Scene },
                locationPathName = Output,
                target = BuildTarget.StandaloneLinux64,
                options = BuildOptions.None,
            });
            if (report.summary.result == BuildResult.Succeeded)
            {
                WriteLauncher();
                Debug.Log($"Game built: {Output} ({report.summary.totalSize / (1024 * 1024)} MB). Start it with {Launcher}");
            }
            else
                Debug.LogError($"Game build failed: {report.summary.result}");
        }
    }
}

using System;
using UnityEngine;

namespace TrackRush
{
    public enum Medal { None, Bronze, Silver, Gold, Author }

    public class LevelData
    {
        public string Name;
        public string Subtitle;
        public float Width = 16f;
        public float AuthorTime = 60f;   // seconds; set from Autopilot runs (Skill 1)
        public Color Grass = new(0.24f, 0.48f, 0.2f);
        public int Seed = 1;
        public Action<TrackBuilder> Layout;

        // Trackmania-style medal thresholds derived from the author time
        public float GoldTime => Mathf.Ceil(AuthorTime * 1.06f);
        public float SilverTime => Mathf.Ceil(AuthorTime * 1.2f);
        public float BronzeTime => Mathf.Ceil(AuthorTime * 1.5f);

        public Medal MedalFor(float time) =>
            time <= 0 ? Medal.None
            : time <= AuthorTime ? Medal.Author
            : time <= GoldTime ? Medal.Gold
            : time <= SilverTime ? Medal.Silver
            : time <= BronzeTime ? Medal.Bronze
            : Medal.None;

        public float TimeFor(Medal m) => m switch
        {
            Medal.Author => AuthorTime,
            Medal.Gold => GoldTime,
            Medal.Silver => SilverTime,
            Medal.Bronze => BronzeTime,
            _ => 0,
        };
    }

    public static class Levels
    {
        public static readonly LevelData[] All =
        {
            new()
            {
                Name = "01  Warm-Up", Subtitle = "Learn the car: turns, a hill and your first boost.",
                AuthorTime = 30.5f, Seed = 11,
                Layout = b => b.Straight(160).Turn(90, 70, 8).Straight(140).Boost(40).Straight(160).Checkpoint()
                    .Turn(-90, 55, 10).Hill(160, 14).Straight(80).Turn(-120, 50, 12).Straight(140).Checkpoint()
                    .Turn(100, 75, 8).Hill(140, -12).Straight(220).Finish(),
            },
            new()
            {
                Name = "02  Banked", Subtitle = "Hairpins with steep banking. Trust the wall of the turn.",
                AuthorTime = 33.8f, Seed = 22, Grass = new Color(0.27f, 0.5f, 0.22f),
                Layout = b => b.Straight(120).Boost(40).Straight(100).Turn(150, 45, 28).Straight(120).Checkpoint()
                    .Turn(-170, 40, 30).Straight(140).Boost(30).Straight(120).Checkpoint()
                    .Turn(120, 50, 25).Hill(150, 16).Turn(-70, 60, 15).Hill(150, -14).Straight(160).Finish(),
            },
            new()
            {
                Name = "03  Airtime", Subtitle = "Three jumps. Keep your speed up or you won't make it.",
                AuthorTime = 24.8f, Seed = 33, Grass = new Color(0.22f, 0.45f, 0.19f),
                Layout = b => b.Straight(120).Boost(40).Hill(140, 16).Straight(30).Jump().Straight(100).Checkpoint()
                    .Turn(110, 55, 10).Straight(60).Boost(30).Straight(40).Jump(28, 7, 44).Straight(90).Hill(100, -8).Checkpoint()
                    .Turn(-100, 50, 10).Straight(60).Boost(30).Straight(40).Jump(34, 8, 48, 60).Straight(90).Hill(120, -12).Straight(60).Finish(),
            },
            new()
            {
                Name = "04  Loop", Subtitle = "Go upside down. Too slow and you'll fall out.",
                AuthorTime = 30.4f, Seed = 44, Grass = new Color(0.25f, 0.47f, 0.21f),
                Layout = b => b.Straight(100).Boost(40).Straight(80).LoopTheLoop(20, 22).Straight(120).Checkpoint()
                    .Turn(-120, 50, 15).Straight(80).Hill(120, 16).Straight(20).Boost(30).Jump().Straight(90).Hill(100, -12).Checkpoint()
                    .Turn(140, 55, 20).Straight(80).Boost(30).Straight(60).LoopTheLoop(22, -24).Straight(160).Finish(),
            },
            new()
            {
                Name = "05  Full Send", Subtitle = "Jumps, loops, banks and boosts. The final exam.",
                AuthorTime = 35.8f, Seed = 55, Grass = new Color(0.2f, 0.42f, 0.18f),
                Layout = b => b.Straight(80).Boost(40).Hill(120, 14).Jump().Straight(100).Checkpoint()
                    .Turn(-130, 45, 25).Straight(80).Boost(30).Straight(60).LoopTheLoop(20, 22).Straight(80).Checkpoint()
                    .Turn(160, 50, 30).Straight(60).Boost(30).Hill(140, 12).Turn(-80, 60, 15).Straight(40).Jump(32, 7, 46).Straight(90).Hill(120, -16).Checkpoint()
                    .Turn(90, 60, 15).Boost(40).Straight(80).LoopTheLoop(24, -26).Straight(140).Finish(),
            },
        };
    }
}

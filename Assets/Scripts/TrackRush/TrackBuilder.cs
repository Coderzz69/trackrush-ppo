using System.Collections.Generic;
using UnityEngine;

namespace TrackRush
{
    /// Lays a track out like a turtle, one metre at a time, from Trackmania-style pieces.
    /// Produces a centreline with a full frame (forward/up) per sample plus feature flags.
    public class TrackBuilder
    {
        public const float Gravity = 20f; // must match CarController.Gravity so jumps land where designed

        public readonly List<Vector3> Pos = new(), Fwd = new(), Up = new();
        public readonly List<bool> GapFlags = new(), BoostFlags = new(), LoopFlags = new();
        public readonly List<int> Checkpoints = new();
        public int FinishIndex = -1;

        Vector3 p = new(0, 0.5f, 0);
        float yaw, pitch, bank; // radians

        Vector3 Forward => new(Mathf.Cos(pitch) * Mathf.Sin(yaw), Mathf.Sin(pitch), Mathf.Cos(pitch) * Mathf.Cos(yaw));
        Vector3 FlatRight => new(Mathf.Cos(yaw), 0, -Mathf.Sin(yaw));

        static float Smooth(float t) => t * t * (3 - 2 * t);

        void Emit(bool gap = false, bool boost = false, bool loop = false)
        {
            var f = Forward;
            var up = Vector3.Cross(f, FlatRight);
            up = Quaternion.AngleAxis(bank * Mathf.Rad2Deg, f) * up;
            Pos.Add(p);
            Fwd.Add(f);
            Up.Add(up);
            GapFlags.Add(gap);
            BoostFlags.Add(boost);
            LoopFlags.Add(loop);
        }

        void Advance(bool boost = false)
        {
            p += Forward;
            Emit(boost: boost);
        }

        // ------------------------------------------------------------------ pieces

        public TrackBuilder Straight(float length)
        {
            for (int i = 0; i < (int)length; i++) Advance();
            return this;
        }

        /// Orange chevrons: driving over them gives a turbo.
        public TrackBuilder Boost(float length)
        {
            for (int i = 0; i < (int)length; i++) Advance(boost: true);
            return this;
        }

        /// Positive degrees turn right. Bank tilts the road into the turn.
        public TrackBuilder Turn(float degrees, float radius, float bankDegrees = 0f)
        {
            int n = Mathf.Max(1, (int)(Mathf.Abs(degrees) * Mathf.Deg2Rad * radius));
            float step = degrees * Mathf.Deg2Rad / n;
            float roll = -Mathf.Sign(degrees) * bankDegrees * Mathf.Deg2Rad; // outside edge goes up
            for (int i = 0; i < n; i++)
            {
                float t = (i + 0.5f) / n;
                bank = roll * Smooth(Mathf.Clamp01(t / 0.25f)) * Smooth(Mathf.Clamp01((1 - t) / 0.25f));
                yaw += step;
                Advance();
            }
            bank = 0;
            return this;
        }

        /// Smooth bump (or dip, with negative height) that changes elevation by `height`.
        public TrackBuilder Hill(float length, float height)
        {
            int n = (int)length;
            float a = height * Mathf.PI / (2f * length);
            for (int i = 0; i < n; i++)
            {
                pitch = Mathf.Asin(Mathf.Clamp(a * Mathf.Sin(Mathf.PI * (i + 0.5f) / n), -0.9f, 0.9f));
                Advance();
            }
            pitch = 0;
            return this;
        }

        public TrackBuilder PitchTo(float length, float degrees)
        {
            float p0 = pitch, p1 = degrees * Mathf.Deg2Rad;
            int n = (int)length;
            for (int i = 0; i < n; i++)
            {
                pitch = Mathf.Lerp(p0, p1, Smooth((i + 1f) / n));
                Advance();
            }
            return this;
        }

        /// No road for `length` metres; the centreline follows the flight path of a car taking
        /// off at `designSpeed` (the slowest speed that still makes it).
        public TrackBuilder Gap(float length, float designSpeed)
        {
            var heading = new Vector3(Mathf.Sin(yaw), 0, Mathf.Cos(yaw));
            float vx = designSpeed * Mathf.Cos(pitch), vy = designSpeed * Mathf.Sin(pitch);
            var start = p;
            for (int i = 1; i <= (int)length; i++)
            {
                float t = i / vx;
                p = start + heading * i + Vector3.up * (vy * t - 0.5f * Gravity * t * t);
                pitch = Mathf.Atan2(vy - Gravity * t, vx);
                Emit(gap: true);
            }
            return this;
        }

        /// Kicker, gap, then a landing slope that eases back to flat. Faster cars land further
        /// down the slope, so the jump forgives anything above designSpeed.
        public TrackBuilder Jump(float gap = 30f, float rampDegrees = 7f, float designSpeed = 45f, float landing = 50f)
        {
            PitchTo(20, rampDegrees);
            Gap(gap, designSpeed);
            return PitchTo(landing, 0);
        }

        /// Full vertical loop; the exit is shifted sideways so it clears the entrance.
        public TrackBuilder LoopTheLoop(float radius, float sideShift)
        {
            int n = (int)(2 * Mathf.PI * radius);
            var right = FlatRight;
            for (int i = 0; i < n; i++)
            {
                pitch += 2 * Mathf.PI / n;
                float k = sideShift * (Smooth((i + 1f) / n) - Smooth((float)i / n));
                p += Forward + right * k;
                Emit(loop: true);
            }
            pitch = 0;
            return this;
        }

        public TrackBuilder Checkpoint()
        {
            Checkpoints.Add(Mathf.Max(0, Pos.Count - 1));
            return this;
        }

        public TrackBuilder Finish(float runoff = 80f)
        {
            FinishIndex = Mathf.Max(0, Pos.Count - 1);
            return Straight(runoff);
        }

        /// Lifts the whole track so its lowest road point sits just above the ground.
        public void Settle(float minHeight = 0.5f)
        {
            float lowest = float.MaxValue;
            for (int i = 0; i < Pos.Count; i++)
                if (!GapFlags[i]) lowest = Mathf.Min(lowest, Pos[i].y);
            float lift = Mathf.Max(0, minHeight - lowest);
            for (int i = 0; i < Pos.Count; i++) Pos[i] += Vector3.up * lift;
        }
    }
}

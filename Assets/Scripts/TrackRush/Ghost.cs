using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace TrackRush
{
    /// A recorded run: car pose sampled at a fixed rate from the start signal.
    public class GhostRun
    {
        public const float Rate = 30f;
        public readonly List<Vector3> Positions = new();
        public readonly List<Quaternion> Rotations = new();

        public float Duration => Positions.Count / Rate;

        public void Record(Transform t)
        {
            Positions.Add(t.position);
            Rotations.Add(t.rotation);
        }

        public bool Sample(float time, out Vector3 pos, out Quaternion rot)
        {
            pos = default;
            rot = Quaternion.identity;
            if (Positions.Count < 2) return false;
            float f = Mathf.Clamp(time * Rate, 0, Positions.Count - 1.001f);
            int i = (int)f;
            float t = f - i;
            pos = Vector3.Lerp(Positions[i], Positions[i + 1], t);
            rot = Quaternion.Slerp(Rotations[i], Rotations[i + 1], t);
            return true;
        }

        static string PathFor(int level) => Path.Combine(Application.persistentDataPath, "ghosts", $"track{level + 1:00}.ghost");

        public void Save(int level)
        {
            var path = PathFor(level);
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            using var w = new BinaryWriter(File.Create(path));
            w.Write(Positions.Count);
            for (int i = 0; i < Positions.Count; i++)
            {
                var p = Positions[i];
                var q = Rotations[i];
                w.Write(p.x); w.Write(p.y); w.Write(p.z);
                w.Write(q.x); w.Write(q.y); w.Write(q.z); w.Write(q.w);
            }
        }

        public static GhostRun Load(int level)
        {
            var path = PathFor(level);
            if (!File.Exists(path)) return null;
            try
            {
                using var r = new BinaryReader(File.OpenRead(path));
                var run = new GhostRun();
                int n = r.ReadInt32();
                for (int i = 0; i < n; i++)
                {
                    run.Positions.Add(new Vector3(r.ReadSingle(), r.ReadSingle(), r.ReadSingle()));
                    run.Rotations.Add(new Quaternion(r.ReadSingle(), r.ReadSingle(), r.ReadSingle(), r.ReadSingle()));
                }
                return run;
            }
            catch (IOException)
            {
                return null;
            }
        }

        public static void DeleteAll()
        {
            var dir = Path.Combine(Application.persistentDataPath, "ghosts");
            if (Directory.Exists(dir)) Directory.Delete(dir, true);
        }
    }
}

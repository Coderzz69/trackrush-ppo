using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace TrackRush
{
    /// A point-to-point stunt track built from a LevelData layout. Answers "where am I on the
    /// track" for cars, the game and RL agents.
    public class Track : MonoBehaviour
    {
        public const float SlabThickness = 0.8f;
        public const float WallHeight = 1.0f;
        public const float AIGrip = 40f;    // lateral accel the speed profile is built for
        public const float AIBrake = 24f;

        /// Every drivable collider: true = road, false = grass.
        public static readonly Dictionary<Collider, bool> Surfaces = new();

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        static void ResetStatics() => Surfaces.Clear();

        public LevelData Level { get; private set; }
        public Vector3[] Points, Forward, Up, Right;
        public float[] Dist, Curvature, TargetSpeed;
        public bool[] IsGap, IsBoost, IsLoop;
        public int[] Checkpoints;
        public int FinishIndex { get; private set; }
        public int StartIndex { get; private set; }
        public int StartLineIndex { get; private set; }
        public float Length { get; private set; }
        public float HalfWidth { get; private set; }
        public Vector3 Center { get; private set; }
        public float Radius { get; private set; }

        Vector3 origin;
        readonly List<Collider> ownColliders = new();

        public static Track Build(LevelData level, Vector3 offset = default, bool scenery = true)
        {
            var t = new GameObject("Track - " + level.Name).AddComponent<Track>();
            t.origin = offset;
            t.Generate(level, scenery);
            return t;
        }

        void OnDestroy()
        {
            foreach (var c in ownColliders) Surfaces.Remove(c);
        }

        void Generate(LevelData lvl, bool scenery)
        {
            Level = lvl;
            HalfWidth = lvl.Width * 0.5f;

            var b = new TrackBuilder();
            lvl.Layout(b);
            b.Settle();
            int n = b.Pos.Count;
            Points = new Vector3[n];
            Forward = b.Fwd.ToArray();
            Up = b.Up.ToArray();
            Right = new Vector3[n];
            Dist = new float[n];
            IsGap = b.GapFlags.ToArray();
            IsBoost = b.BoostFlags.ToArray();
            IsLoop = b.LoopFlags.ToArray();
            for (int i = 0; i < n; i++)
            {
                Points[i] = b.Pos[i] + origin;
                Right[i] = Vector3.Cross(Up[i], Forward[i]).normalized;
                if (i > 0) Dist[i] = Dist[i - 1] + Vector3.Distance(Points[i], Points[i - 1]);
            }
            Length = Dist[n - 1];
            Checkpoints = b.Checkpoints.ToArray();
            FinishIndex = b.FinishIndex >= 0 ? b.FinishIndex : n - 1;
            StartLineIndex = IndexAt(12f);
            StartIndex = IndexAt(4f);

            Vector3 min = Points[0], max = Points[0];
            foreach (var p in Points) { min = Vector3.Min(min, p); max = Vector3.Max(max, p); }
            Center = new Vector3((min.x + max.x) * 0.5f, 0, (min.z + max.z) * 0.5f);
            Radius = Mathf.Max(max.x - min.x, max.z - min.z) * 0.5f;

            ComputeSpeedProfile();
            BuildGround(min, max);
            BuildRoad();
            BuildBoostPads();
            BuildLinesAndGates();
            BuildPillars();
            if (scenery) BuildScenery(new System.Random(lvl.Seed));
        }

        void ComputeSpeedProfile()
        {
            int n = Points.Length;
            Curvature = new float[n];
            TargetSpeed = new float[n];
            for (int i = 0; i < n; i++)
            {
                int a = Mathf.Max(0, i - 3), c = Mathf.Min(n - 1, i + 3);
                float ds = Mathf.Max(Dist[c] - Dist[a], 0.01f);
                Curvature[i] = Vector3.Dot(Forward[c] - Forward[a], Right[i]) / ds; // sideways turn rate in the road plane
                TargetSpeed[i] = Mathf.Min(CarController.BoostTopSpeed, Mathf.Sqrt(AIGrip / Mathf.Max(Mathf.Abs(Curvature[i]), 1e-4f)));
            }
            for (int i = n - 2; i >= 0; i--)
            {
                float ds = Dist[i + 1] - Dist[i];
                TargetSpeed[i] = Mathf.Min(TargetSpeed[i], Mathf.Sqrt(TargetSpeed[i + 1] * TargetSpeed[i + 1] + 2f * AIBrake * ds));
            }
        }

        // ---------------------------------------------------------------- queries

        public int IndexAt(float distance)
        {
            distance = Mathf.Clamp(distance, 0, Length);
            int lo = 0, hi = Dist.Length - 1;
            while (lo < hi)
            {
                int mid = (lo + hi + 1) / 2;
                if (Dist[mid] <= distance) lo = mid; else hi = mid - 1;
            }
            return lo;
        }

        public void SampleAt(float distance, out Vector3 pos, out Vector3 fwd, out Vector3 up)
        {
            int i = IndexAt(distance);
            int j = Mathf.Min(i + 1, Points.Length - 1);
            float seg = Dist[j] - Dist[i];
            float t = seg > 1e-4f ? Mathf.Clamp01((Mathf.Clamp(distance, 0, Length) - Dist[i]) / seg) : 0;
            pos = Vector3.Lerp(Points[i], Points[j], t);
            fwd = Vector3.Slerp(Forward[i], Forward[j], t);
            up = Vector3.Slerp(Up[i], Up[j], t);
        }

        public int FindNearest(Vector3 pos, int hint, int window = 30)
        {
            int n = Points.Length, best = 0;
            float bd = float.MaxValue;
            if (hint >= 0)
            {
                for (int i = Mathf.Max(0, hint - window); i <= Mathf.Min(n - 1, hint + window); i++)
                {
                    float d = (Points[i] - pos).sqrMagnitude;
                    if (d < bd) { bd = d; best = i; }
                }
                float limit = HalfWidth + 25f;
                if (bd < limit * limit) return best;
                bd = float.MaxValue;
            }
            for (int i = 0; i < n; i++)
            {
                float d = (Points[i] - pos).sqrMagnitude;
                if (d < bd) { bd = d; best = i; }
            }
            return best;
        }

        /// Updates the car's track index, offsets and progress. Returns metres gained this call.
        public float Locate(CarController c)
        {
            int i = FindNearest(c.Position, c.TrackIndex);
            c.TrackIndex = i;
            Vector3 d = c.Position - Points[i];
            c.LateralOffset = Vector3.Dot(d, Right[i]);
            c.HeightAboveTrack = Vector3.Dot(d, Up[i]);
            c.OnRoad = !IsGap[i] && Mathf.Abs(c.LateralOffset) < HalfWidth + 0.5f && c.HeightAboveTrack > -1.5f && c.HeightAboveTrack < 6f;
            float progress = Mathf.Clamp(Dist[i] + Vector3.Dot(d, Forward[i]), 0, Length);
            float delta = progress - c.Progress;
            c.Progress = progress;
            return delta;
        }

        /// True when the car is physically inside a gate's frame (not flying over it or on the grass beside it).
        public bool InGate(CarController c) =>
            Mathf.Abs(c.LateralOffset) < HalfWidth + 3f && c.HeightAboveTrack > -3f && c.HeightAboveTrack < 12f;

        /// Standing start pose, or a respawn pose at any sample.
        public Pose PoseAt(int index) =>
            new(Points[index] + Up[index] * 0.15f, Quaternion.LookRotation(Forward[index], Up[index]));

        // ---------------------------------------------------------------- meshes

        class MeshBuilder
        {
            readonly List<Vector3> v = new(), nrm = new();
            readonly List<Vector2> uv = new();
            readonly List<int> tris = new();

            /// Quad with corners [across][along]; normals given per along-edge.
            public void Quad(Vector3 a0, Vector3 b0, Vector3 a1, Vector3 b1, Vector3 n0, Vector3 n1,
                Vector2 uvA0, Vector2 uvB0, Vector2 uvA1, Vector2 uvB1)
            {
                int k = v.Count;
                v.Add(a0); v.Add(b0); v.Add(a1); v.Add(b1);
                nrm.Add(n0); nrm.Add(n0); nrm.Add(n1); nrm.Add(n1);
                uv.Add(uvA0); uv.Add(uvB0); uv.Add(uvA1); uv.Add(uvB1);
                bool flip = Vector3.Dot(Vector3.Cross(a1 - a0, b0 - a0), n0 + n1) < 0;
                if (!flip) { tris.Add(k); tris.Add(k + 2); tris.Add(k + 1); tris.Add(k + 2); tris.Add(k + 3); tris.Add(k + 1); }
                else { tris.Add(k); tris.Add(k + 1); tris.Add(k + 2); tris.Add(k + 2); tris.Add(k + 1); tris.Add(k + 3); }
            }

            public Mesh Build()
            {
                var m = new Mesh { indexFormat = IndexFormat.UInt32 };
                m.SetVertices(v);
                m.SetNormals(nrm);
                m.SetUVs(0, uv);
                m.SetTriangles(tris, 0);
                m.RecalculateBounds();
                return m;
            }
        }

        GameObject MeshObject(string name, Mesh mesh, Material mat, bool collider = false, bool? surface = null)
        {
            var go = new GameObject(name);
            go.transform.SetParent(transform, false);
            go.AddComponent<MeshFilter>().sharedMesh = mesh;
            go.AddComponent<MeshRenderer>().sharedMaterial = mat;
            if (collider)
            {
                var mc = go.AddComponent<MeshCollider>();
                mc.sharedMesh = mesh;
                ownColliders.Add(mc);
                if (surface.HasValue) Surfaces[mc] = surface.Value;
            }
            return go;
        }

        bool Segment(int i) => i >= 0 && i + 1 < Points.Length && !IsGap[i] && !IsGap[i + 1];

        void BuildGround(Vector3 min, Vector3 max)
        {
            var tex = MatUtil.MakeTexture(64, 64, (x, y) =>
            {
                float stripe = x < 32 ? 1.06f : 0.94f;
                float noise = 0.92f + 0.16f * Mathf.PerlinNoise(x * 0.31f, y * 0.29f);
                return new Color(stripe * noise, stripe * noise, stripe * noise);
            });
            float size = Mathf.Max(max.x - min.x, max.z - min.z) + 2000f;
            var mat = MatUtil.MakeLit(Level.Grass, 0.1f, 0f, tex);
            mat.SetTextureScale("_BaseMap", new Vector2(size / 24f, size / 24f));
            var g = GameObject.CreatePrimitive(PrimitiveType.Plane);
            g.name = "Ground";
            g.transform.SetParent(transform, false);
            g.transform.position = Center;
            g.transform.localScale = new Vector3(size / 10f, 1, size / 10f);
            g.GetComponent<MeshRenderer>().sharedMaterial = mat;
            var col = g.GetComponent<Collider>();
            ownColliders.Add(col);
            Surfaces[col] = false;
        }

        void BuildRoad()
        {
            var roadTex = MatUtil.MakeTexture(128, 128, (x, y) =>
            {
                float u = (x + 0.5f) / 128f;
                if ((u > 0.02f && u < 0.04f) || (u > 0.96f && u < 0.98f)) return new Color(0.92f, 0.92f, 0.92f);
                float g = 0.24f + 0.05f * Mathf.PerlinNoise(x * 0.7f, y * 0.7f);
                return new Color(g, g, g * 1.05f);
            });
            var wallTex = MatUtil.MakeTexture(2, 2, (x, y) => x == 0 ? new Color(0.85f, 0.1f, 0.12f) : new Color(0.95f, 0.95f, 0.95f), FilterMode.Point);

            var top = new MeshBuilder();
            var under = new MeshBuilder();
            var walls = new MeshBuilder();
            float hw = HalfWidth, th = SlabThickness;
            const float wallT = 0.4f;

            for (int i = 0; i + 1 < Points.Length; i++)
            {
                if (!Segment(i)) continue;
                int j = i + 1;
                Vector3 Ri = Right[i], Rj = Right[j], Ui = Up[i], Uj = Up[j];
                Vector3 li = Points[i] - Ri * hw, ri = Points[i] + Ri * hw;
                Vector3 lj = Points[j] - Rj * hw, rj = Points[j] + Rj * hw;
                float vi = Dist[i] / Level.Width, vj = Dist[j] / Level.Width;

                top.Quad(li, ri, lj, rj, Ui, Uj, new(0, vi), new(1, vi), new(0, vj), new(1, vj));

                // slab underside and edges
                under.Quad(li - Ui * th, ri - Ui * th, lj - Uj * th, rj - Uj * th, -Ui, -Uj, default, default, default, default);
                under.Quad(li - Ui * th, li, lj - Uj * th, lj, -Ri, -Rj, default, default, default, default);
                under.Quad(ri, ri - Ui * th, rj, rj - Uj * th, Ri, Rj, default, default, default, default);
                if (!Segment(i - 1)) // start cap
                    under.Quad(li - Ui * th, ri - Ui * th, li, ri, -Forward[i], -Forward[i], default, default, default, default);
                if (!Segment(j)) // end cap
                    under.Quad(lj, rj, lj - Uj * th, rj - Uj * th, Forward[j], Forward[j], default, default, default, default);

                // edge walls (inner face, outer face, top)
                float ui = Dist[i] / 4f, uj = Dist[j] / 4f;
                foreach (float s in new[] { -1f, 1f })
                {
                    Vector3 bi = Points[i] + Ri * (s * hw), bj = Points[j] + Rj * (s * hw);
                    Vector3 oi = bi + Ri * (s * wallT), oj = bj + Rj * (s * wallT);
                    Vector3 hi = Ui * WallHeight, hj = Uj * WallHeight;
                    walls.Quad(bi, bi + hi, bj, bj + hj, -Ri * s, -Rj * s, new(ui, 0), new(ui, 1), new(uj, 0), new(uj, 1));
                    walls.Quad(oi + hi, oi, oj + hj, oj, Ri * s, Rj * s, new(ui, 1), new(ui, 0), new(uj, 1), new(uj, 0));
                    walls.Quad(bi + hi, oi + hi, bj + hj, oj + hj, Ui, Uj, new(ui, 1), new(ui, 1), new(uj, 1), new(uj, 1));
                }
            }

            MeshObject("Road", top.Build(), MatUtil.MakeLit(Color.white, 0.3f, 0f, roadTex), collider: true, surface: true);
            MeshObject("Road Slab", under.Build(), MatUtil.MakeLit(new Color(0.55f, 0.56f, 0.6f), 0.2f));
            MeshObject("Walls", walls.Build(), MatUtil.MakeLit(Color.white, 0.4f, 0f, wallTex)); // walls are enforced by CarController
        }

        void BuildBoostPads()
        {
            // chevrons pointing along +v (driving direction)
            var tex = MatUtil.MakeTexture(64, 64, (x, y) =>
            {
                float u = Mathf.Abs((x + 0.5f) / 64f - 0.5f) * 2f;
                float v = (y + 0.5f) / 64f;
                bool chevron = Mathf.Repeat(v - u * 0.35f, 0.5f) < 0.22f;
                return chevron ? new Color(1f, 0.85f, 0.2f) : new Color(1f, 0.35f, 0.05f);
            }, FilterMode.Bilinear);
            var mat = MatUtil.MakeUnlit(Color.white);
            mat.SetTexture("_BaseMap", tex);
            var mb = new MeshBuilder();
            float w = HalfWidth * 0.7f;
            bool any = false;
            for (int i = 0; i + 1 < Points.Length; i++)
            {
                if (!IsBoost[i] || !IsBoost[i + 1]) continue;
                any = true;
                int j = i + 1;
                Vector3 lift = Up[i] * 0.04f, liftJ = Up[j] * 0.04f;
                float vi = Dist[i] / 8f, vj = Dist[j] / 8f;
                mb.Quad(Points[i] - Right[i] * w + lift, Points[i] + Right[i] * w + lift,
                    Points[j] - Right[j] * w + liftJ, Points[j] + Right[j] * w + liftJ,
                    Up[i], Up[j], new(0, vi), new(1, vi), new(0, vj), new(1, vj));
            }
            if (any) MeshObject("Boost Pads", mb.Build(), mat);
        }

        void Strip(string name, int index, float length, Material mat)
        {
            var mb = new MeshBuilder();
            int end = IndexAt(Dist[index] + length);
            for (int i = index; i < end; i++)
            {
                int j = i + 1;
                float vi = (Dist[i] - Dist[index]) / length, vj = (Dist[j] - Dist[index]) / length;
                Vector3 li = Up[i] * 0.05f, lj = Up[j] * 0.05f;
                mb.Quad(Points[i] - Right[i] * HalfWidth + li, Points[i] + Right[i] * HalfWidth + li,
                    Points[j] - Right[j] * HalfWidth + lj, Points[j] + Right[j] * HalfWidth + lj,
                    Up[i], Up[j], new(0, vi), new(1, vi), new(0, vj), new(1, vj));
            }
            MeshObject(name, mb.Build(), mat);
        }

        void Gate(string name, int index, Color color, Material beamMat = null)
        {
            var root = new GameObject(name).transform;
            root.SetParent(transform, false);
            root.SetPositionAndRotation(Points[index], Quaternion.LookRotation(Forward[index], Up[index]));
            var frame = MatUtil.MakeLit(new Color(0.12f, 0.12f, 0.14f), 0.5f, 0.4f);
            var glow = MatUtil.MakeUnlit(color * 2.5f);
            float x = HalfWidth + 1.4f;
            foreach (float s in new[] { -1f, 1f })
            {
                MatUtil.Prim(PrimitiveType.Cube, root, new Vector3(s * x, 4f, 0), new Vector3(1.2f, 8f, 1.2f), frame);
                MatUtil.Prim(PrimitiveType.Cube, root, new Vector3(s * (x - 0.62f), 4f, 0), new Vector3(0.1f, 7f, 0.5f), glow);
            }
            MatUtil.Prim(PrimitiveType.Cube, root, new Vector3(0, 8.4f, 0), new Vector3(2 * x + 1.2f, 1.6f, 1.2f), beamMat ?? frame);
            MatUtil.Prim(PrimitiveType.Cube, root, new Vector3(0, 8.4f, -0.62f), new Vector3(2 * x - 1f, 0.5f, 0.05f), glow);
        }

        void BuildLinesAndGates()
        {
            var checker = MatUtil.MakeTexture(16, 2, (x, y) => ((x + y) % 2 == 0) ? Color.white : Color.black, FilterMode.Point, false);
            var checkerMat = MatUtil.MakeLit(Color.white, 0.3f, 0f, checker);
            Strip("Start Line", StartLineIndex, 2f, checkerMat);
            Strip("Finish Line", FinishIndex, 3f, checkerMat);

            var bannerMat = MatUtil.MakeLit(Color.white, 0.3f, 0f,
                MatUtil.MakeTexture(32, 4, (x, y) => ((x + y) % 2 == 0) ? Color.white : Color.black, FilterMode.Point, false));
            Gate("Start Gate", StartLineIndex, new Color(0.2f, 1f, 0.3f));
            for (int k = 0; k < Checkpoints.Length; k++) Gate($"Checkpoint {k + 1}", Checkpoints[k], new Color(0.2f, 0.6f, 1f));
            Gate("Finish Gate", FinishIndex, new Color(1f, 0.25f, 0.2f), bannerMat);
        }

        void BuildPillars()
        {
            var concrete = MatUtil.MakeLit(new Color(0.6f, 0.6f, 0.63f), 0.2f);
            var pillars = new List<CombineInstance>();
            var cube = MatUtil.PrimitiveMesh(PrimitiveType.Cube);
            for (int i = 6; i < Points.Length; i += 14)
            {
                if (!Segment(i) || IsLoop[i] || Up[i].y < 0.8f) continue;
                float bottom = Points[i].y - SlabThickness;
                if (bottom < 1.5f) continue;
                var flat = Vector3.ProjectOnPlane(Forward[i], Vector3.up);
                var rot = flat.sqrMagnitude > 1e-4f ? Quaternion.LookRotation(flat) : Quaternion.identity;
                pillars.Add(new CombineInstance
                {
                    mesh = cube,
                    transform = Matrix4x4.TRS(new Vector3(Points[i].x, bottom * 0.5f, Points[i].z), rot, new Vector3(3f, bottom, 2f)),
                });
            }
            if (pillars.Count == 0) return;
            var m = new Mesh { indexFormat = IndexFormat.UInt32 };
            m.CombineMeshes(pillars.ToArray());
            MeshObject("Pillars", m, concrete);
        }

        // ---------------------------------------------------------------- scenery

        float FlatDistanceToTrack(Vector3 p)
        {
            float best = float.MaxValue;
            for (int i = 0; i < Points.Length; i += 2)
            {
                float dx = Points[i].x - p.x, dz = Points[i].z - p.z;
                best = Mathf.Min(best, dx * dx + dz * dz);
            }
            return Mathf.Sqrt(best);
        }

        Transform Place(string model, float scale, Vector3 pos, Vector3 faceDir, float clearance)
        {
            pos.y = 0;
            if (FlatDistanceToTrack(pos) < clearance) return null;
            var t = KenneyAssets.Spawn(model, transform, scale);
            if (t == null) return null;
            faceDir.y = 0;
            // Kenney buildings face -Z; flip so the open side looks at the track
            t.SetPositionAndRotation(pos, Quaternion.LookRotation(faceDir.sqrMagnitude > 1e-4f ? faceDir : Vector3.forward) * Quaternion.Euler(0, 180f, 0));
            return t;
        }

        void BuildScenery(System.Random rng)
        {
            float R() => (float)rng.NextDouble();
            float clear = HalfWidth + 6f;

            // grandstands along the start and finish straights
            foreach (int anchor in new[] { StartLineIndex, FinishIndex })
                for (int k = -2; k <= 2; k++)
                {
                    int i = IndexAt(Dist[anchor] + k * 18f);
                    var flatRight = Vector3.ProjectOnPlane(Right[i], Vector3.up).normalized;
                    foreach (float s in new[] { -1f, 1f })
                        Place(k % 2 == 0 ? "grandStandCovered" : "grandStand", 1.7f,
                            Points[i] + flatRight * s * (HalfWidth + 16f), -flatRight * s, clear + 6f);
                }
            {
                var flatRight = Vector3.ProjectOnPlane(Right[StartLineIndex], Vector3.up).normalized;
                Place("bannerTowerRed", 1f, Points[StartLineIndex] + flatRight * (HalfWidth + 6f), -flatRight, clear - 2f);
                Place("bannerTowerGreen", 1f, Points[StartLineIndex] - flatRight * (HalfWidth + 6f), flatRight, clear - 2f);
            }

            // billboards and tents outside the corners
            int last = -1000;
            for (int i = 0; i < Points.Length; i++)
            {
                if (Mathf.Abs(Curvature[i]) < 1f / 90f || i - last < 90 || IsLoop[i]) continue;
                last = i;
                var outside = -Vector3.ProjectOnPlane(Right[i], Vector3.up).normalized * Mathf.Sign(Curvature[i]);
                var basePos = Points[i] + outside * (HalfWidth + 14f);
                Place(R() < 0.6f ? "billboard" : "billboardLow", 1.3f, basePos, -outside, clear + 3f);
                Place(R() < 0.5f ? "tent" : "tentLong", 0.9f, basePos + outside * 18f, -outside, clear + 10f);
            }

            float extent = Radius + 250f;
            int placed = 0;
            for (int attempt = 0; attempt < 1500 && placed < 300; attempt++)
            {
                var p = new Vector3(Center.x + (R() * 2 - 1) * extent, 0, Center.z + (R() * 2 - 1) * extent);
                if (Place(R() < 0.5f ? "treeLarge" : "treeSmall", 0.6f + R() * 0.5f, p,
                        Quaternion.Euler(0, R() * 360f, 0) * Vector3.forward, clear + 4f) != null)
                    placed++;
            }
        }
    }
}

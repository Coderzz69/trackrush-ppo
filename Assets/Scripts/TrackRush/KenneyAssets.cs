using System.Collections.Generic;
using UnityEngine;

namespace TrackRush
{
    /// Loads Kenney Racing Kit models (CC0, kenney.nl) from Resources/KenneyRacing and normalizes them:
    /// pivot at bottom-center, uniform scale, punchier colours. Returns null if an asset is missing,
    /// so callers can fall back to primitives.
    public static class KenneyAssets
    {
        const string Folder = "KenneyRacing/";

        static readonly Dictionary<string, Material> remapped = new();

        // Play mode can start without a domain reload, so statics outlive the materials they cache.
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        static void ResetCache() => remapped.Clear();

        // The kit's material colours import washed-out in linear colour space; swap in stronger ones.
        static readonly Dictionary<string, Color> palette = new()
        {
            { "red", new Color(0.82f, 0.1f, 0.1f) },
            { "green", new Color(0.15f, 0.55f, 0.3f) },
            { "grass", new Color(0.18f, 0.45f, 0.2f) },
            { "bark", new Color(0.45f, 0.32f, 0.2f) },
            { "carTire", new Color(0.08f, 0.08f, 0.09f) },
            { "road", new Color(0.25f, 0.25f, 0.27f) },
            { "grey", new Color(0.88f, 0.88f, 0.9f) },
        };

        public static bool Has(string name) => Resources.Load<GameObject>(Folder + name) != null;

        /// Instantiates a model with its footprint centered on the returned root's origin and its base at y=0.
        public static Transform Spawn(string name, Transform parent, float scale, bool flip = false)
        {
            var prefab = Resources.Load<GameObject>(Folder + name);
            if (prefab == null) return null;

            var root = new GameObject(name).transform;
            var turn = new GameObject("Orient").transform;
            turn.SetParent(root, false);
            var inst = Object.Instantiate(prefab, turn, false).transform;

            var b = WorldBounds(inst);
            inst.position -= new Vector3(b.center.x, b.min.y, b.center.z);
            if (flip) turn.localRotation = Quaternion.Euler(0, 180f, 0);
            root.localScale = Vector3.one * scale;
            root.SetParent(parent, false);
            Recolor(inst, null, default);
            return root;
        }

        public static Bounds WorldBounds(Transform t)
        {
            var rs = t.GetComponentsInChildren<Renderer>();
            if (rs.Length == 0) return new Bounds(t.position, Vector3.zero);
            var b = rs[0].bounds;
            foreach (var r in rs) b.Encapsulate(r.bounds);
            return b;
        }

        /// Swaps materials to the stronger palette; materials named in `tint` get that colour instead.
        public static void Recolor(Transform t, HashSet<string> tintNames, Color tint, Color? secondary = null)
        {
            foreach (var r in t.GetComponentsInChildren<Renderer>())
            {
                var mats = r.sharedMaterials;
                for (int i = 0; i < mats.Length; i++)
                {
                    var m = mats[i];
                    if (m == null) continue;
                    string key = m.name.Replace(" (Instance)", "");
                    if (tintNames != null && tintNames.Contains(key))
                        mats[i] = MatUtil.MakeLit(tint, 0.8f, 0.35f);
                    else if (secondary.HasValue && key == "grey")
                        mats[i] = MatUtil.MakeLit(secondary.Value, 0.7f, 0.3f);
                    else if (palette.TryGetValue(key, out var c))
                    {
                        if (!remapped.TryGetValue(key, out var shared) || shared == null)
                        {
                            shared = new Material(m);
                            shared.SetColor("_BaseColor", c);
                            remapped[key] = shared;
                        }
                        mats[i] = shared;
                    }
                }
                r.sharedMaterials = mats;
            }
        }

        static readonly HashSet<string> carBodyColours = new() { "red", "green", "orange", "grey", "white" };

        /// Kenney race car, recoloured for a team. Returns the visual root plus wheel spin/steer nodes.
        public static Transform SpawnCar(Transform parent, Team team, List<Transform> steer, List<Transform> spin)
        {
            var prefab = Resources.Load<GameObject>(Folder + "raceCarRed");
            if (prefab == null) return null;

            // 13.5 units long in the kit; scale to ~5.4 m and slim the width a little
            var root = new GameObject("Kenney Car").transform;
            var turn = new GameObject("Orient").transform;
            turn.SetParent(root, false);
            var inst = Object.Instantiate(prefab, turn, false).transform;
            var b = WorldBounds(inst);
            inst.position -= new Vector3(b.center.x, b.min.y, b.center.z);
            turn.localRotation = Quaternion.Euler(0, 180f, 0); // model faces -Z

            // body gets the team colour; wheel rims get the secondary colour
            foreach (Transform child in inst)
            {
                bool wheel = child.name.StartsWith("wheel");
                if (!wheel) Recolor(child, new HashSet<string> { "red" }, team.Primary);
                else Recolor(child, null, default, team.Secondary);
            }

            // wrap each wheel: steer pivot -> spin node -> mesh, all centered on the wheel
            foreach (var wheel in new List<Transform>(inst.GetComponentsInChildren<Transform>()))
            {
                if (!wheel.name.StartsWith("wheel")) continue;
                var c = WorldBounds(wheel).center;
                var pivot = new GameObject(wheel.name + " Pivot").transform;
                pivot.SetParent(wheel.parent, false);
                pivot.SetPositionAndRotation(c, root.rotation); // car-aligned axes, not the flipped model's
                var spinNode = new GameObject("Spin").transform;
                spinNode.SetParent(pivot, false);
                wheel.SetParent(spinNode, true);
                if (wheel.name.Contains("Front")) steer.Add(pivot);
                spin.Add(spinNode);
            }

            root.localScale = new Vector3(0.32f, 0.4f, 0.4f);
            root.SetParent(parent, false);
            return root;
        }
    }
}

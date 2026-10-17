using System.Collections.Generic;
using UnityEngine;

namespace TrackRush
{
    public class Team
    {
        public string Name;
        public Color Primary, Secondary;

        public Team(string name, Color primary, Color secondary)
        {
            Name = name;
            Primary = primary;
            Secondary = secondary;
        }
    }

    public static class Teams
    {
        public static readonly Team Player = new("Your Team", new Color(1f, 0.82f, 0.05f), new Color(0.08f, 0.08f, 0.1f));

        public static readonly Team[] AI =
        {
            new("Rosso Veloce", new Color(0.85f, 0.05f, 0.05f), new Color(1f, 0.85f, 0.1f)),
            new("Argento", new Color(0.72f, 0.75f, 0.78f), new Color(0f, 0.75f, 0.65f)),
            new("Toro Blu", new Color(0.08f, 0.12f, 0.35f), new Color(0.95f, 0.1f, 0.15f)),
            new("Papaya GP", new Color(1f, 0.5f, 0.05f), new Color(0.1f, 0.3f, 0.9f)),
            new("Verde Racing", new Color(0f, 0.4f, 0.3f), new Color(0.8f, 1f, 0.2f)),
            new("Bleu Rosa", new Color(0.1f, 0.4f, 0.85f), new Color(1f, 0.4f, 0.7f)),
            new("Lupo Grigio", new Color(0.92f, 0.92f, 0.92f), new Color(0.8f, 0.05f, 0.1f)),
            new("Oxford Blue", new Color(0.05f, 0.2f, 0.55f), new Color(0.9f, 0.9f, 1f)),
            new("Neon Lime", new Color(0.3f, 0.9f, 0.2f), new Color(0.05f, 0.05f, 0.05f)),
            new("Midnight", new Color(0.15f, 0.15f, 0.2f), new Color(0.3f, 0.5f, 1f)),
        };

        public static readonly (string Name, string Abbrev)[] Drivers =
        {
            ("Luca Moreau", "MOR"), ("Kenji Tanaka", "TAN"), ("Rafael Silva", "SIL"), ("Jonas Becker", "BEC"),
            ("Anya Novak", "NOV"), ("David Okafor", "OKA"), ("Marco Bellini", "BEL"), ("Sven Lindqvist", "LIN"),
            ("Pablo Garcia", "GAR"), ("Tom Walsh", "WAL"),
        };
    }

    /// Builds the stunt car (Kenney model, primitive fallback) and wires up physics.
    public static class CarFactory
    {
        static readonly Dictionary<Color, Material> boxMaterials = new();

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        static void ResetCache() => boxMaterials.Clear();

        /// One shared material per colour, so 1000 box cars don't mean 1000 materials.
        static Material BoxMaterial(Color c)
        {
            if (!boxMaterials.TryGetValue(c, out var m) || m == null) boxMaterials[c] = m = MatUtil.MakeLit(c, 0.5f);
            return m;
        }

        public static CarController Create(Team team, string driver, string abbrev, bool isPlayer, bool lightweight = false)
        {
            var root = new GameObject($"Car - {abbrev}");
            var steer = new List<Transform>();
            var spin = new List<Transform>();
            if (lightweight)
                MatUtil.Prim(PrimitiveType.Cube, root.transform, new Vector3(0, 0.6f, 0), new Vector3(1.9f, 0.7f, 5f), BoxMaterial(team.Primary));
            else if (KenneyAssets.SpawnCar(root.transform, team, steer, spin) == null)
                BuildPrimitiveBody(root.transform, team, steer, spin);

            // kinematic and collider-free: CarController moves the car itself (see its FixedUpdate)
            var rb = root.AddComponent<Rigidbody>();
            rb.isKinematic = true;
            rb.useGravity = false;
            rb.interpolation = RigidbodyInterpolation.Interpolate;
            if (lightweight)
                foreach (var r in root.GetComponentsInChildren<Renderer>()) r.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;

            // keep the car's own probes (and other cars') from hitting car colliders
            foreach (var t in root.GetComponentsInChildren<Transform>(true)) t.gameObject.layer = CarController.CarLayer;

            var car = root.AddComponent<CarController>();
            car.Setup(driver, abbrev, team, isPlayer, steer.ToArray(), spin.ToArray());

            if (isPlayer)
            {
                var engine = new GameObject("Engine Audio");
                engine.transform.SetParent(root.transform, false);
                engine.AddComponent<AudioSource>();
                engine.AddComponent<EngineAudio>().Car = car;
            }
            return car;
        }

        /// Visual-only translucent car that replays a recorded run.
        public static Transform CreateGhost(Color tint)
        {
            var root = new GameObject("Ghost");
            var steer = new List<Transform>();
            var spin = new List<Transform>();
            if (KenneyAssets.SpawnCar(root.transform, Teams.Player, steer, spin) == null)
                BuildPrimitiveBody(root.transform, Teams.Player, steer, spin);
            var mat = MatUtil.MakeTransparentUnlit(tint);
            foreach (var r in root.GetComponentsInChildren<Renderer>())
            {
                var mats = r.sharedMaterials;
                for (int i = 0; i < mats.Length; i++) mats[i] = mat;
                r.sharedMaterials = mats;
                r.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            }
            return root.transform;
        }

        /// Fallback car built from primitives (used when the Kenney models aren't in the project).
        static void BuildPrimitiveBody(Transform root, Team team, List<Transform> steer, List<Transform> spin)
        {
            var body = new GameObject("Body").transform;
            body.SetParent(root, false);

            var prim = MatUtil.MakeLit(team.Primary, 0.8f, 0.35f);
            var sec = MatUtil.MakeLit(team.Secondary, 0.7f, 0.2f);
            var carbon = MatUtil.MakeLit(new Color(0.05f, 0.05f, 0.06f), 0.55f, 0.2f);
            var tyre = MatUtil.MakeLit(new Color(0.04f, 0.04f, 0.04f), 0.15f);
            var visor = MatUtil.MakeLit(new Color(0.05f, 0.05f, 0.05f), 0.95f, 0.6f);

            void Box(Vector3 pos, Vector3 size, Material m, Vector3 euler = default) =>
                MatUtil.Prim(PrimitiveType.Cube, body, pos, size, m, euler);

            Box(new Vector3(0, 0.12f, -0.1f), new Vector3(1.5f, 0.08f, 4.4f), carbon);               // floor
            Box(new Vector3(0, 0.42f, 0.4f), new Vector3(0.75f, 0.45f, 2.4f), prim);                  // monocoque
            Box(new Vector3(0, 0.33f, 2.25f), new Vector3(0.36f, 0.22f, 1.5f), prim, new Vector3(4, 0, 0)); // nose
            Box(new Vector3(0, 0.25f, 3.0f), new Vector3(0.3f, 0.15f, 0.3f), sec);                    // nose tip
            Box(new Vector3(0, 0.12f, 2.85f), new Vector3(1.95f, 0.05f, 0.5f), sec);                  // front wing
            Box(new Vector3(0, 0.2f, 2.65f), new Vector3(1.8f, 0.05f, 0.3f), prim, new Vector3(-15, 0, 0));
            foreach (float sx in new[] { -1f, 1f })
            {
                Box(new Vector3(0.98f * sx, 0.18f, 2.8f), new Vector3(0.04f, 0.22f, 0.6f), carbon);   // FW endplate
                Box(new Vector3(0.62f * sx, 0.38f, -0.1f), new Vector3(0.55f, 0.42f, 1.7f), prim);    // sidepod
                Box(new Vector3(0.62f * sx, 0.42f, 0.76f), new Vector3(0.5f, 0.3f, 0.05f), carbon);   // sidepod inlet
                Box(new Vector3(0.53f * sx, 0.82f, -2.3f), new Vector3(0.04f, 0.55f, 0.75f), carbon); // RW endplate
                Box(new Vector3(0.5f * sx, 0.38f, 1.75f), new Vector3(0.75f, 0.04f, 0.06f), carbon);  // front suspension
                Box(new Vector3(0.5f * sx, 0.4f, -1.55f), new Vector3(0.7f, 0.04f, 0.06f), carbon);   // rear suspension
                Box(new Vector3(0.24f * sx, 0.92f, 0.05f), new Vector3(0.05f, 0.05f, 0.7f), carbon);  // halo sides
            }
            Box(new Vector3(0, 0.62f, -0.9f), new Vector3(0.55f, 0.5f, 1.6f), prim);                  // engine cover
            Box(new Vector3(0, 0.95f, -0.35f), new Vector3(0.3f, 0.35f, 0.4f), carbon);               // airbox
            Box(new Vector3(0, 0.95f, -1.3f), new Vector3(0.04f, 0.35f, 1.2f), prim);                 // shark fin
            Box(new Vector3(0, 0.95f, -2.3f), new Vector3(1.05f, 0.05f, 0.45f), sec);                 // rear wing
            Box(new Vector3(0, 1.06f, -2.42f), new Vector3(1.05f, 0.05f, 0.25f), prim, new Vector3(-20, 0, 0));
            Box(new Vector3(0, 0.62f, -2.15f), new Vector3(0.06f, 0.5f, 0.1f), carbon);               // RW pillar
            Box(new Vector3(0, 0.2f, -2.3f), new Vector3(1f, 0.2f, 0.4f), carbon);                    // diffuser
            Box(new Vector3(0, 0.66f, 0.05f), new Vector3(0.5f, 0.06f, 0.8f), carbon);                // cockpit
            Box(new Vector3(0, 0.84f, 0.42f), new Vector3(0.05f, 0.18f, 0.05f), carbon);              // halo strut
            MatUtil.Prim(PrimitiveType.Sphere, body, new Vector3(0, 0.8f, -0.05f), Vector3.one * 0.32f, sec); // helmet
            MatUtil.Prim(PrimitiveType.Cube, body, new Vector3(0, 0.82f, 0.08f), new Vector3(0.22f, 0.07f, 0.1f), visor);

            foreach (var (x, z, front) in new[] { (-0.88f, 1.75f, true), (0.88f, 1.75f, true), (-0.86f, -1.55f, false), (0.86f, -1.55f, false) })
            {
                float radius = front ? 0.36f : 0.38f;
                var pivot = new GameObject("Wheel").transform;
                pivot.SetParent(body, false);
                pivot.localPosition = new Vector3(x, radius, z);
                var spinNode = new GameObject("Spin").transform;
                spinNode.SetParent(pivot, false);
                MatUtil.Prim(PrimitiveType.Cylinder, spinNode, Vector3.zero, new Vector3(radius * 2, front ? 0.19f : 0.23f, radius * 2), tyre, new Vector3(0, 0, 90));
                // coloured rim stripe so the spin is visible
                MatUtil.Prim(PrimitiveType.Cube, spinNode, new Vector3(Mathf.Sign(x) * (front ? 0.19f : 0.23f), 0, 0), new Vector3(0.02f, radius * 1.1f, 0.12f), sec);
                if (front) steer.Add(pivot);
                spin.Add(spinNode);
            }
        }
    }
}

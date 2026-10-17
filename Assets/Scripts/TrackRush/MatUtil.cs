using UnityEngine;

namespace TrackRush
{
    public static class MatUtil
    {
        static Shader _lit, _unlit;
        static Shader LitShader => _lit ? _lit : (_lit = Shader.Find("Universal Render Pipeline/Lit"));
        static Shader UnlitShader => _unlit ? _unlit : (_unlit = Shader.Find("Universal Render Pipeline/Unlit"));

        public static Material MakeLit(Color c, float smoothness = 0.35f, float metallic = 0f, Texture tex = null)
        {
            var m = new Material(LitShader);
            m.SetColor("_BaseColor", c);
            m.SetFloat("_Smoothness", smoothness);
            m.SetFloat("_Metallic", metallic);
            if (tex != null) m.SetTexture("_BaseMap", tex);
            m.enableInstancing = true;
            return m;
        }

        public static Material MakeUnlit(Color c)
        {
            // builds strip shaders nothing references; fall back to Lit rather than crash
            var m = new Material(UnlitShader != null ? UnlitShader : LitShader);
            m.SetColor("_BaseColor", c);
            return m;
        }

        /// URP Unlit set up for alpha blending (used for the ghost car).
        public static Material MakeTransparentUnlit(Color c)
        {
            var m = MakeUnlit(c);
            m.SetFloat("_Surface", 1f);
            m.SetFloat("_Blend", 0f);
            m.SetFloat("_SrcBlend", (float)UnityEngine.Rendering.BlendMode.SrcAlpha);
            m.SetFloat("_DstBlend", (float)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            m.SetFloat("_ZWrite", 0f);
            m.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
            m.SetOverrideTag("RenderType", "Transparent");
            m.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
            return m;
        }

        public static Texture2D MakeTexture(int w, int h, System.Func<int, int, Color> f,
            FilterMode filter = FilterMode.Bilinear, bool mips = true)
        {
            var t = new Texture2D(w, h, TextureFormat.RGBA32, mips);
            var px = new Color[w * h];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    px[y * w + x] = f(x, y);
            t.SetPixels(px);
            t.filterMode = filter;
            t.wrapMode = TextureWrapMode.Repeat;
            t.anisoLevel = 8;
            t.Apply(mips);
            return t;
        }

        public static Texture2D Solid(Color c) => MakeTexture(1, 1, (x, y) => c, FilterMode.Point, false);

        public static Texture2D Circle(int size)
        {
            float r = size * 0.5f;
            return MakeTexture(size, size, (x, y) =>
            {
                float d = Vector2.Distance(new Vector2(x + 0.5f, y + 0.5f), new Vector2(r, r));
                return new Color(1, 1, 1, Mathf.Clamp01(r - d));
            }, FilterMode.Bilinear, false);
        }

        /// Creates a primitive without a collider, parented and placed in local space.
        public static GameObject Prim(PrimitiveType type, Transform parent, Vector3 pos, Vector3 scale, Material mat,
            Vector3 euler = default)
        {
            var g = GameObject.CreatePrimitive(type);
            Object.DestroyImmediate(g.GetComponent<Collider>());
            g.transform.SetParent(parent, false);
            g.transform.localPosition = pos;
            g.transform.localEulerAngles = euler;
            g.transform.localScale = scale;
            g.GetComponent<MeshRenderer>().sharedMaterial = mat;
            return g;
        }

        public static Mesh PrimitiveMesh(PrimitiveType type)
        {
            var g = GameObject.CreatePrimitive(type);
            var mesh = g.GetComponent<MeshFilter>().sharedMesh;
            Object.DestroyImmediate(g);
            return mesh;
        }
    }
}

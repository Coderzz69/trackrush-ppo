using UnityEngine;

namespace TrackRush
{
    /// Procedural engine note (no audio assets). Needs an AudioSource on the same GameObject.
    public class EngineAudio : MonoBehaviour
    {
        public CarController Car;

        AudioSource source;
        int sampleRate;
        double phase, phaseSub;
        float freq = 150f, amp, lowpass;
        volatile float targetFreq = 150f, targetAmp;
        readonly System.Random rng = new();

        void Start()
        {
            sampleRate = AudioSettings.outputSampleRate;
            source = GetComponent<AudioSource>();
            source.playOnAwake = false;
            source.spatialBlend = 0f;
            source.volume = 0.3f;
            source.loop = true;
            source.Play();
        }

        void Update()
        {
            if (Car == null) return;
            targetFreq = 140f + 560f * Car.RpmFraction;
            bool silent = Time.timeScale == 0f || !Car.isActiveAndEnabled;
            targetAmp = silent ? 0f : 0.35f + 0.65f * Car.Controls.Throttle;
        }

        void OnAudioFilterRead(float[] data, int channels)
        {
            if (sampleRate == 0) return;
            for (int i = 0; i < data.Length; i += channels)
            {
                freq += (targetFreq - freq) * 0.0005f;
                amp += (targetAmp - amp) * 0.0005f;
                phase += freq / sampleRate;
                if (phase >= 1) phase -= 1;
                phaseSub += freq * 0.5 / sampleRate;
                if (phaseSub >= 1) phaseSub -= 1;
                float saw = (float)(phase * 2 - 1);
                float sq = phaseSub < 0.5 ? 1f : -1f;
                float noise = (float)(rng.NextDouble() * 2 - 1);
                float s = saw * 0.5f + sq * 0.3f + noise * 0.08f;
                lowpass += (s - lowpass) * 0.3f;
                float o = lowpass * amp * 0.5f;
                for (int c = 0; c < channels; c++) data[i + c] = o;
            }
        }
    }
}

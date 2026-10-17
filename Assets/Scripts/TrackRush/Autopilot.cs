using UnityEngine;

namespace TrackRush
{
    /// Scripted driver: aims at a point ahead on the centreline and follows the track's speed
    /// profile. Used to measure author times and to test tracks; not part of normal play.
    [DefaultExecutionOrder(-50)]
    public class Autopilot : MonoBehaviour
    {
        public float Skill = 1f;
        CarController car;

        void Awake() => car = GetComponent<CarController>();

        void FixedUpdate()
        {
            var t = car.Track;
            if (car.Frozen || t == null || car.TrackIndex < 0)
            {
                car.Controls = default;
                return;
            }
            float speed = car.ForwardSpeed;
            t.SampleAt(car.Progress + 8f + Mathf.Abs(speed) * 0.4f, out var target, out _, out _);
            var local = Quaternion.Inverse(car.Rotation) * (target - car.Position);
            float ang = Mathf.Atan2(local.x, local.z) * Mathf.Rad2Deg;

            float tv = t.TargetSpeed[t.IndexAt(car.Progress + 4f)] * Skill;
            var c = new CarInput { Steer = car.Grounded ? Mathf.Clamp(ang / 12f, -1f, 1f) : 0f };
            if (!car.Grounded || speed < tv - 1f) c.Throttle = 1f;
            else if (speed < tv) c.Throttle = 0.4f;
            if (car.Grounded && speed > tv + 2f) c.Brake = Mathf.Clamp01((speed - tv) / 6f);
            car.Controls = c;
        }
    }
}

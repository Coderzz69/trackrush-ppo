using UnityEngine;
using UnityEngine.InputSystem;

namespace TrackRush
{
    public class ChaseCamera : MonoBehaviour
    {
        public enum Mode { Orbit, Chase, FarChase, Cockpit, Free }

        public Mode CamMode = Mode.Orbit;
        public CarController Target;

        Vector3 orbitCenter;
        float orbitRadius, orbitAngle;
        Vector3 smoothFwd = Vector3.forward, smoothUp = Vector3.up;
        Camera cam;
        float freeYaw, freePitch, freeSpeed = 40f;

        void Awake() => cam = GetComponent<Camera>();

        public void SetOrbit(Vector3 center, float radius)
        {
            orbitCenter = center;
            orbitRadius = radius;
            CamMode = Mode.Orbit;
            Target = null;
        }

        public void Follow(CarController car)
        {
            Target = car;
            smoothFwd = car.transform.forward;
            smoothUp = car.transform.up;
            if (CamMode == Mode.Orbit || CamMode == Mode.Free) CamMode = Mode.Chase;
        }

        /// Fly-cam starting from wherever the camera is now: hold right mouse to look, WASD to move,
        /// Q/E down/up, Shift faster, scroll wheel changes the base speed.
        public void SetFree()
        {
            var e = transform.eulerAngles;
            freeYaw = e.y;
            freePitch = e.x > 180f ? e.x - 360f : e.x;
            transform.rotation = Quaternion.Euler(freePitch, freeYaw, 0);
            CamMode = Mode.Free;
            Target = null;
        }

        public void CycleMode()
        {
            if (Target == null) return;
            CamMode = CamMode switch
            {
                Mode.Chase => Mode.FarChase,
                Mode.FarChase => Mode.Cockpit,
                _ => Mode.Chase,
            };
        }

        void LateUpdate()
        {
            float dt = Time.unscaledDeltaTime;
            if (CamMode == Mode.Free)
            {
                UpdateFree(dt);
                return;
            }
            if (Cursor.lockState != CursorLockMode.None)
            {
                Cursor.lockState = CursorLockMode.None;
                Cursor.visible = true;
            }
            if (CamMode == Mode.Orbit || Target == null)
            {
                orbitAngle += 5f * dt;
                var offset = Quaternion.Euler(0, orbitAngle, 0) * new Vector3(0, orbitRadius * 0.75f, -orbitRadius * 1.25f);
                transform.position = orbitCenter + offset;
                transform.LookAt(orbitCenter);
                cam.fieldOfView = 55f;
                return;
            }

            var tr = Target.transform;
            float speed01 = Mathf.Clamp01(Target.SpeedKmh / 380f);
            if (CamMode == Mode.Cockpit)
            {
                transform.SetPositionAndRotation(tr.TransformPoint(0, 1.3f, -0.5f), tr.rotation * Quaternion.Euler(4f, 0, 0));
                cam.fieldOfView = Mathf.Lerp(70f, 85f, speed01);
                return;
            }

            // follow the road's "up" through banks and loops, world up while flying
            var wantUp = Target.Grounded ? tr.up : Vector3.up;
            smoothUp = Vector3.Slerp(smoothUp, wantUp, 1f - Mathf.Exp(-5f * dt)).normalized;
            smoothFwd = Vector3.Slerp(smoothFwd, tr.forward, 1f - Mathf.Exp(-7f * dt)).normalized;
            bool far = CamMode == Mode.FarChase;
            float dist = far ? 11f : 7f, height = far ? 4f : 2.4f;
            transform.position = tr.position - smoothFwd * dist + smoothUp * height;
            transform.rotation = Quaternion.LookRotation(tr.position + smoothFwd * 6f + smoothUp * 1f - transform.position, smoothUp);
            cam.fieldOfView = Mathf.Lerp(62f, 80f, speed01);
        }

        void UpdateFree(float dt)
        {
            var kb = Keyboard.current;
            var mouse = Mouse.current;
            cam.fieldOfView = 65f;
            if (mouse != null)
            {
                bool look = mouse.rightButton.isPressed;
                Cursor.lockState = look ? CursorLockMode.Locked : CursorLockMode.None;
                Cursor.visible = !look;
                if (look)
                {
                    var d = mouse.delta.ReadValue();
                    freeYaw += d.x * 0.15f;
                    freePitch = Mathf.Clamp(freePitch - d.y * 0.15f, -89f, 89f);
                    transform.rotation = Quaternion.Euler(freePitch, freeYaw, 0);
                }
                float scroll = mouse.scroll.ReadValue().y;
                if (scroll != 0) freeSpeed = Mathf.Clamp(freeSpeed * (scroll > 0 ? 1.2f : 1f / 1.2f), 5f, 600f);
            }
            if (kb == null) return;

            var move = Vector3.zero;
            if (kb.wKey.isPressed || kb.upArrowKey.isPressed) move += transform.forward;
            if (kb.sKey.isPressed || kb.downArrowKey.isPressed) move -= transform.forward;
            if (kb.dKey.isPressed || kb.rightArrowKey.isPressed) move += transform.right;
            if (kb.aKey.isPressed || kb.leftArrowKey.isPressed) move -= transform.right;
            if (kb.eKey.isPressed) move += Vector3.up;
            if (kb.qKey.isPressed) move -= Vector3.up;
            float speed = freeSpeed * (kb.leftShiftKey.isPressed ? 4f : 1f);
            transform.position += move.normalized * speed * dt;
        }

        void OnDisable()
        {
            Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;
        }
    }
}

// OrbitCamera.cs — small Play-mode camera controller for inspecting the demo surface.
// Hold LEFT mouse button and drag to orbit around the target; scroll wheel to zoom.
// Not part of the algorithm — purely a verification convenience.

using UnityEngine;

namespace ReconGridDC.Demo
{
    public sealed class OrbitCamera : MonoBehaviour
    {
        public Vector3 target   = Vector3.zero;
        public float   distance = 13f;
        public float   xSpeed   = 4f;
        public float   ySpeed   = 4f;
        public float   yMin     = -85f;
        public float   yMax     = 85f;
        public float   zoomSpeed = 8f;
        public float   minDist  = 1f;
        public float   maxDist  = 200f;

        float x, y;

        void Start()
        {
            Vector3 ang = transform.eulerAngles;
            x = ang.y;
            y = ang.x;
        }

        void LateUpdate()
        {
            if (Input.GetMouseButton(0))   // hold left mouse button to orbit
            {
                x += Input.GetAxis("Mouse X") * xSpeed;
                y -= Input.GetAxis("Mouse Y") * ySpeed;
                y = Mathf.Clamp(y, yMin, yMax);
            }

            float scroll = Input.GetAxis("Mouse ScrollWheel");
            distance = Mathf.Clamp(distance - scroll * zoomSpeed, minDist, maxDist);

            Quaternion rot = Quaternion.Euler(y, x, 0f);
            transform.rotation = rot;
            transform.position = rot * new Vector3(0f, 0f, -distance) + target;
        }
    }
}

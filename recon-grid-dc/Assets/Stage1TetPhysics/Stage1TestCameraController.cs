using UnityEngine;

namespace ReconGridDC.Stage1TetPhysics
{
    [DisallowMultipleComponent]
    public sealed class Stage1TestCameraController : MonoBehaviour
    {
        public float moveSpeed = 4f;
        public float lookSensitivity = 2.5f;
        [Header("Camera Zoom")]
        [Min(0.01f)]
        [Tooltip("World-space distance moved by one mouse-wheel increment. Start around 0.25 and adjust for the current scene scale.")]
        public float scrollSpeed = 4f;
        [Tooltip("Allow WASD/arrow-key camera translation. Disable this in tool-control test scenes so those keys exclusively control the active tool.")]
        public bool enableKeyboardTranslation = true;

        float _yaw;
        float _pitch;

        void Awake()
        {
            Vector3 angles = transform.eulerAngles;
            _yaw = angles.y;
            _pitch = angles.x;
        }

        void Update()
        {
            if (Input.GetMouseButton(1))
            {
                _yaw += Input.GetAxis("Mouse X") * lookSensitivity;
                _pitch = Mathf.Clamp(_pitch - Input.GetAxis("Mouse Y") * lookSensitivity, -85f, 85f);
                transform.rotation = Quaternion.Euler(_pitch, _yaw, 0f);
            }

            if (enableKeyboardTranslation)
            {
                Vector3 input = new Vector3(Input.GetAxisRaw("Horizontal"), 0f, Input.GetAxisRaw("Vertical"));
                if (input.sqrMagnitude > 1f) input.Normalize();
                float speed = moveSpeed * (Input.GetKey(KeyCode.LeftShift) ? 2f : 1f);
                transform.position += transform.TransformDirection(input) * speed * Time.unscaledDeltaTime;
            }
            // Scroll input is reported as a per-frame wheel delta; scrollSpeed is distance per wheel increment.
            transform.position += transform.forward * Input.mouseScrollDelta.y * scrollSpeed;
        }
    }
}

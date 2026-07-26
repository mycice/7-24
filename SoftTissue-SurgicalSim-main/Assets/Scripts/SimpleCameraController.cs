#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

using UnityEngine;

namespace UnityTemplateProjects
{
    public class SimpleCameraController : MonoBehaviour
    {
        class CameraState
        {
            public float yaw;
            public float pitch;
            public float roll;
            public float x;
            public float y;
            public float z;

            public void SetFromTransform(Transform t)
            {
                pitch = NormalizeAngle(t.eulerAngles.x);
                yaw = t.eulerAngles.y;
                roll = t.eulerAngles.z;
                x = t.position.x;
                y = t.position.y;
                z = t.position.z;
            }

            static float NormalizeAngle(float angle)
            {
                return angle > 180f ? angle - 360f : angle;
            }

            public void Translate(Vector3 translation)
            {
                Vector3 rotatedTranslation = Quaternion.Euler(pitch, yaw, roll) * translation;

                x += rotatedTranslation.x;
                y += rotatedTranslation.y;
                z += rotatedTranslation.z;
            }

            public void LerpTowards(CameraState target, float positionLerpPct, float rotationLerpPct)
            {
                yaw = Mathf.Lerp(yaw, target.yaw, rotationLerpPct);
                pitch = Mathf.Lerp(pitch, target.pitch, rotationLerpPct);
                roll = Mathf.Lerp(roll, target.roll, rotationLerpPct);

                x = Mathf.Lerp(x, target.x, positionLerpPct);
                y = Mathf.Lerp(y, target.y, positionLerpPct);
                z = Mathf.Lerp(z, target.z, positionLerpPct);
            }

            public void UpdateTransform(Transform t)
            {
                t.eulerAngles = new Vector3(pitch, yaw, roll);
                t.position = new Vector3(x, y, z);
            }
        }

        const float k_MouseSensitivityMultiplier = 0.01f;

        CameraState m_TargetCameraState = new CameraState();
        CameraState m_InterpolatingCameraState = new CameraState();

        [Header("Movement Settings")]
        [Tooltip("Camera pan distance per mouse movement unit.")]
        public float panSpeed = 0.025f;

        [Tooltip("Distance moved along the view direction per mouse wheel step.")]
        public float zoomSpeed = 0.35f;

        [Tooltip("Time it takes to interpolate camera position 99% of the way to the target."), Range(0.001f, 1f)]
        public float positionLerpTime = 0.2f;

        [Header("Rotation Settings")]
        [Tooltip("Multiplier for the sensitivity of the rotation.")]
        public float mouseSensitivity = 60.0f;

        [Tooltip("X = Change in mouse position.\nY = Multiplicative factor for camera rotation.")]
        public AnimationCurve mouseSensitivityCurve = new AnimationCurve(new Keyframe(0f, 0.5f, 0f, 5f), new Keyframe(1f, 2.5f, 0f, 0f));

        [Tooltip("Time it takes to interpolate camera rotation 99% of the way to the target."), Range(0.001f, 1f)]
        public float rotationLerpTime = 0.01f;

        [Tooltip("Whether or not to invert our Y axis for mouse input to rotation.")]
        public bool invertY = false;

#if ENABLE_INPUT_SYSTEM
        InputAction lookAction;
        InputAction zoomAction;

        void Start()
        {
            var map = new InputActionMap("Simple Camera Controller");

            lookAction = map.AddAction("look", binding: "<Mouse>/delta");
            zoomAction = map.AddAction("Zoom", binding: "<Mouse>/scroll");

            lookAction.AddBinding("<Gamepad>/rightStick").WithProcessor("scaleVector2(x=15, y=15)");
            lookAction.Enable();
            zoomAction.Enable();
        }

#endif

        void OnEnable()
        {
            m_TargetCameraState.SetFromTransform(transform);
            m_InterpolatingCameraState.SetFromTransform(transform);
        }


        void Update()
        {
            // Exit Sample

            if (IsEscapePressed())
            {
                Application.Quit();
#if UNITY_EDITOR
                UnityEditor.EditorApplication.isPlaying = false;
#endif
            }

            // Hide and lock the cursor while rotating or panning.
            if (IsLeftMouseButtonDown() || IsRightMouseButtonDown())
            {
                Cursor.visible = false;
                Cursor.lockState = CursorLockMode.Locked;
            }

            if ((IsLeftMouseButtonUp() || IsRightMouseButtonUp()) && !IsCameraDragAllowed())
            {
                Cursor.visible = true;
                Cursor.lockState = CursorLockMode.None;
            }

            // Rotation
            if (IsCameraRotationAllowed())
            {
                var mouseMovement = GetInputLookRotation() * k_MouseSensitivityMultiplier * mouseSensitivity;
                if (invertY)
                    mouseMovement.y = -mouseMovement.y;

                var mouseSensitivityFactor = mouseSensitivityCurve.Evaluate(mouseMovement.magnitude);

                m_TargetCameraState.yaw += mouseMovement.x * mouseSensitivityFactor;
                m_TargetCameraState.pitch -= mouseMovement.y * mouseSensitivityFactor;
                m_TargetCameraState.pitch = Mathf.Clamp(m_TargetCameraState.pitch, -89f, 89f);
            }

            // Right-drag pans the camera in screen space; the wheel moves along the view direction.
            var translation = Vector3.forward * GetZoomInput() * zoomSpeed;
            if (IsCameraPanAllowed())
            {
                var panDelta = GetInputLookRotation();
                translation += new Vector3(-panDelta.x, -panDelta.y, 0f) * panSpeed;
            }

            m_TargetCameraState.Translate(translation);

            // Framerate-independent interpolation
            // Calculate the lerp amount, such that we get 99% of the way to our target in the specified time
            var positionLerpPct = 1f - Mathf.Exp((Mathf.Log(1f - 0.99f) / positionLerpTime) * Time.deltaTime);
            var rotationLerpPct = 1f - Mathf.Exp((Mathf.Log(1f - 0.99f) / rotationLerpTime) * Time.deltaTime);
            m_InterpolatingCameraState.LerpTowards(m_TargetCameraState, positionLerpPct, rotationLerpPct);

            m_InterpolatingCameraState.UpdateTransform(transform);
        }

        float GetZoomInput()
        {
#if ENABLE_INPUT_SYSTEM
            return zoomAction.ReadValue<Vector2>().y / 120f;
#else
            return Input.mouseScrollDelta.y;
#endif
        }

        Vector2 GetInputLookRotation()
        {
            // try to compensate the diff between the two input systems by multiplying with empirical values
#if ENABLE_INPUT_SYSTEM
            var delta = lookAction.ReadValue<Vector2>();
            delta *= 0.5f; // Account for scaling applied directly in Windows code by old input system.
            delta *= 0.1f; // Account for sensitivity setting on old Mouse X and Y axes.
            return delta;
#else
            return new Vector2(Input.GetAxis("Mouse X"), Input.GetAxis("Mouse Y"));
#endif
        }


        bool IsEscapePressed()
        {
#if ENABLE_INPUT_SYSTEM
            return Keyboard.current != null ? Keyboard.current.escapeKey.isPressed : false;
#else
            return Input.GetKey(KeyCode.Escape);
#endif
        }

        bool IsCameraRotationAllowed()
        {
#if ENABLE_INPUT_SYSTEM
            bool canRotate = Mouse.current != null && Mouse.current.leftButton.isPressed;
            canRotate |= Gamepad.current != null && Gamepad.current.rightStick.ReadValue().magnitude > 0;
            return canRotate;
#else
            return Input.GetMouseButton(0);
#endif
        }

        bool IsCameraPanAllowed()
        {
#if ENABLE_INPUT_SYSTEM
            return Mouse.current != null && Mouse.current.rightButton.isPressed;
#else
            return Input.GetMouseButton(1);
#endif
        }

        bool IsCameraDragAllowed()
        {
            return IsCameraRotationAllowed() || IsCameraPanAllowed();
        }

        bool IsLeftMouseButtonDown()
        {
#if ENABLE_INPUT_SYSTEM
            return Mouse.current != null && Mouse.current.leftButton.wasPressedThisFrame;
#else
            return Input.GetMouseButtonDown(0);
#endif
        }

        bool IsLeftMouseButtonUp()
        {
#if ENABLE_INPUT_SYSTEM
            return Mouse.current != null && Mouse.current.leftButton.wasReleasedThisFrame;
#else
            return Input.GetMouseButtonUp(0);
#endif
        }

        bool IsRightMouseButtonDown()
        {
#if ENABLE_INPUT_SYSTEM
            return Mouse.current != null && Mouse.current.rightButton.wasPressedThisFrame;
#else
            return Input.GetMouseButtonDown(1);
#endif
        }

        bool IsRightMouseButtonUp()
        {
#if ENABLE_INPUT_SYSTEM
            return Mouse.current != null && Mouse.current.rightButton.wasReleasedThisFrame;
#else
            return Input.GetMouseButtonUp(1);
#endif
        }
    }
}

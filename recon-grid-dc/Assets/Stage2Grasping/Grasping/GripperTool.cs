// GripperTool.cs 闁?Da Vinci 濡炲瀛╅悧鍛婂緞閸︻厼鐒荤€规悶鍎遍崣鍧楁晬閸垺鍩傞悗?OBJ 婵☆垪鈧磭鈧兘鎮ч崼顒傜
// 濞达綀娉曢弫?3 濞戞搩浜ｉ～瀣喆?OBJ + 3 濞戞搩浜為～顐﹀箻?OBJ
// Phase 1: 闁规潙鍟跨敮鍥亹閵忕姴缍?(Poking) 闁?缁炬壆澧楅幐鎺旂磾閹寸偟澹愬☉鎾愁槼椤銇?vs 閺夌儐鍨紞瀣垝閹烘垹鎽?
// Phase 2: 闁硅В鏅滈幗婵囧緞閻熸澘绲?(Grasping) 闁?InvMass 闂佸じ绀侀悾鎯р枖?//
// 闂佹鍠氬ú蹇涘箳瑜嶉崺妤呮晬濞?H闁愁偅澹? G/B闁愁偅澹? V/N闁愁偅濡? 1/3闁愁偅甯楀Λ鍡樻姜? 0闁愁偅甯掔槐鎴﹀触?// Runs independently from the soft-body collision solver.

using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;
using ReconGridDC.Stage1TetPhysics.Core;
using ReconGridDC.Stage1TetPhysics.Physics;

namespace ReconGridDC.Stage2Grasping.Grasping
{
    public class GripperTool : MonoBehaviour
    {
        const float FourXModelScale = 0.01f;
        const float FourXCollisionMargin = FourXModelScale * 1.5f;
        const float FourXJawCapsuleRadius = FourXModelScale * 1.2f;
        const float FourXShaftCapsuleRadius = FourXModelScale * 2.8f;

        [Header("OBJ 婵☆垪鈧磭鈧兘寮崶锔筋偨")]
        public string jawUpVisual   = "Stage2Grasping/Haptic_grasper_jaws_up.obj";
        public string jawDownVisual = "Stage2Grasping/Haptic_grasper_jaws_down.obj";
        public string shaftVisual   = "Stage2Grasping/Haptic_grasper_shaft.obj";
        public string jawUpCol      = "Stage2Grasping/Haptic_grasper_jaws_up_collision.obj";
        public string jawDownCol    = "Stage2Grasping/Haptic_grasper_jaws_down_collision.obj";
        public string shaftCol      = "Stage2Grasping/Haptic_grasper_shaft_collision.obj";

        [Header("婵☆垪鈧磭鈧兘宕ｉ崒娑欐")]
        [Tooltip("OBJ unit to world scale. Current preset keeps the enlarged visual gripper at 0.01.")]
        public float modelScale = FourXModelScale;

        [Tooltip("Apply the calibrated enlarged visual/collision gripper preset at runtime.")]
        public bool useFourXGripperPreset = true;

#if false
        [Tooltip("濞戞挶鍊撻柌婊勫緞閸︻厼鐒婚柣鈺冾焾椤曨噣寮堕崱姘剁叐缂備焦娲樺﹢浼村捶?Z 閺夌偛顕▓鎴犫偓鐟邦槼椤ュ﹪寮€ｎ厽绁悷娆愬笒鐎规娊濡?)]
#endif
        public float jawRollAngle = 90f;

#if false
        [Tooltip("婵絽绻嬮柌婊勫緞閸︻厼鐒婚柛鎺戞閸╁棛绱掗弴鈥虫闂婎剦鍋嗛弮閬嶅触閹存粏鍘煫鍥у暢闁伴亶鎯冮崟顔兼閺夌儐鍓濋～妤佹償閿旇　鍋?)]
#if false
        public float jawSelfRollAngle = 90f;

        [Header("闁绘せ鏅濋幃濠囧矗閸屾稒娈?)]
        [Tooltip("缁炬壆澧楅幐?AABB 濠㈣埖鐗楁晶鎸庢綇绾懐鐛?(m)")]
        public float collisionMargin = FourXCollisionMargin;

        [Tooltip("Jaw capsule contact radius (m). Kept close to the visual jaw thickness to avoid oversized dents.")]
        public float capsuleRadius = FourXJawCapsuleRadius;

        [Tooltip("闁哄妫滈棅鈺呮嚄鐠虹儤鎶勫ù锝嗘尭瀹曟劕顕?(m)闁挎稑鐭傚Σ璇差潰閵忊剝缍勯棅顒夊亞閳规稑螣?)]
        public float shaftRadius = FourXShaftCapsuleRadius;

        [Range(3, 10)]
        [Tooltip("闁活潿鍔嬬花顒勫箯閻旈攱鍊ゅ☉鎾卞€撻柌婊勫緞閸︻厼鐒婚柣銊ュ閸忓矂宕舵繝浣虹Ъ闁诡剛绮弳鐔煎Υ閸屾凹娈為柡渚€顣︾槐浼村箮婵犲偆妯嬮柛鎴ｆ濞?1 濞戞搩浜滈崹搴ｇ磼濞嗗海鐟愬Λ鐗堢啲缁遍亶寮堕崱姘剁叐闁告娲滅€?1 濞戞搩浜ｉ崗宀勫炊婵犱胶绉奸柕?)]
        public int jawCapsuleCount = 10;

        [Range(0.25f, 1.5f)]
        [Tooltip("濞寸姴楠搁妵娆撴偉?collision OBJ 闁告帒妫欓宀勫箯閻旈攱鍊ら柛鎴ｆ濞堟垿宕℃繝鍌滅獮缂傚倵鏅滈弬浣哄寲缂佹ɑ娈堕柕?)]
        public float jawCapsuleFitRadiusScale = 0.75f;

        [Tooltip("闁哄嫬澧介妵姘卞枈閻楀牊瀵柤瀹犳硾濞夘厽鎷?Gizmo")]
        public bool showCapsuleGizmo = true;

        [Range(-0.5f, 0.5f)]
        [Tooltip("Move each jaw toward the center line along tool-local X (m).")]
        public float jawHorizontalCenterOffset = 0.012f;

        [Range(-0.5f, 0.5f)]
        [Tooltip("Move each jaw toward the center line along tool-local Y (m).")]
        public float jawVerticalCenterOffset = 0.012f;

        [Range(0f, 5f)]
        [Tooltip("Jaw-only contact friction multiplier. Shaft friction stays zero.")]
        public float jawContactFrictionMultiplier = 1000f;

        [Header("鐎殿喒鍋撻柛?)]
        [Tooltip("闁哄牃鍋撳鍫嗗啰鐐婄€殿喒鍋撻悷娆愬笒鐎?(閹?")]
        public float maxOpenAngle = 50f;
        [Tooltip("鐎殿喒鍋撻柛姘墣椤鏌呴悢宄邦唺 (閹?缂?")]
        public float openCloseSpeed = 60f;

        [Header("濡ゅ倹蓱閺屽娑甸鍌氼唺闁哄鍠愭慨鍕矗?)]
        [Min(0f)]
        [Tooltip("缁绢収鍓涚€规娊寮堕悢鍝ュ闊洤鍟柈銊╂惞鎼粹€崇稉闁哄倻鎳撻幃婊嗐亹閵忊€崇亣闁汇劌瀚伴悵顕€寮鐐寸暦閻犙呭厴閻濐喗鎯?(m)闁?)]
        public float gaussianHeight = 0.02f;

        [Range(0.1f, 1f)]
        [Tooltip("濡ゅ倹蓱閺屽寮介崶褍娅欑€瑰壊鍠氬ù澶屸偓浣冾潐婵嫰宕ｉ弽褍闅橀柛鈺冨枎瀹曟劕顕ラ崟顓熺暠婵絾鏌х欢銉╂晬濞戞粎楔濠㈠爢宥囩闁哄洦鐓″鎵惥婵犲倿鎸紓鍌涙尪閳?)]
        public float gaussianWidth = 0.5f;

        [Range(0.1f, 1f)]
        [Tooltip("鐟滅増甯婄粩鎾礌閺嶃劊浜堕柛锕€妫楅崬瀵告偖椤愩倐鈧牠鏌ㄦ担鍝ユ毎闁汇劌瀚悧瀹犵疀閸愩劌纾圭€垫澘瀚哥槐杈ㄥ緞閺嵮勭函缂侇喗甯掗悺娆愮┍濠靛洤鐦柛鏂诲妽閳ь兛绶ょ槐婵嬫偨閸楃偠鍓ㄩ柟顑嫮婀撮悷娆欑到濞呮帡鎳涢鍡楀Ё閺夆晛娲﹀ù顕€濡?)]
        public float gaussianHardCoreRadius = 0.55f;

        [Min(0f)]
        [Tooltip("濞寸姴楠哥敮顐ｆ叏鐎ｎ亣鍩岄柣妯挎硾闁解晛顭ㄩ幋锝囩畺婵炴挴鈧啿鐓傚Δ鍌浬戦弻澶庛亹閵忋垹笑闁圭鍋撻梻鍥ｅ亾闁哄啫鐖煎Λ?(s)闁挎稑鐭侀鏇熺▔?0 閻炴稏鍔庨妵姘辩博鐎ｎ亜绁柟瀛樺姇閼镐即濡?)]
        public float gaussianFormDuration = 0.15f;

        [Min(1)]
        [Tooltip("Max FixedUpdate attempts after the jaw first reaches the grasp angle.")]
        public int captureRetryPhysicsSteps = 5;

        [Header("闁硅矇鍐ㄧ厬")]
        public float moveSpeed = 0.3f;
        public float rotateSpeed = 60f;
        public float boostMultiplier = 3.0f;

        [Header("闁告瑯鍨甸～瀣礌?)]
        public Color toolColor = new Color(0.7f, 0.7f, 0.78f, 1f);

        // 闁冲厜鍋撻柍鍏夊亾 闁稿浚鍓欑槐鎴︽偐閼哥鍋?闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?
        public bool IsGrasping { get; private set; }
        public float CurrentAngle => _currentAngle;
        public bool HasControlInput { get; private set; }
        public bool WantsToolContact => HasControlInput || _wantClose || IsGrasping;

        // 闁冲厜鍋撻柍鍏夊亾 缂佸鐒﹀﹢渚€鎮╅懜纰樺亾?闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?
        TetMeshData       _data;
        XPBDSolverGPU     _solver;
        TetMeshVisualizer _visualizer;

        Vector3 _toolPos;
        float _toolRotY;
        float _currentAngle;
        bool    _wantClose;
        bool    _initialized;

        // 閻熸瑥妫滈～?GameObjects
        [Header("闁革妇鍎ゅ▍娆徫熼垾宕団偓椋庘偓鐢殿攰閽?)]
        [Tooltip("闁革妇鍎ゅ▍娆愮▔椤撱埄鏆曢柛蹇撶墕閸ㄥ崬顕欓搹瑙勭暠濞戞挸锕ら妵娆撴偉椤忓嫮鎽嶉柣妞绘櫃缂嶅濡?)]
        public GameObject upperJawObject;

        [Tooltip("闁革妇鍎ゅ▍娆愮▔椤撱埄鏆曢柛蹇撶墕閸ㄥ崬顕欓搹瑙勭暠濞戞挸顑呴妵娆撴偉椤忓嫮鎽嶉柣妞绘櫃缂嶅濡?)]
        public GameObject lowerJawObject;

        [Tooltip("闁革妇鍎ゅ▍娆愮▔椤撱埄鏆曢柛蹇撶墕閸ㄥ崬顕欓搹瑙勭暠濠㈣泛婀遍崺鍛村级閸℃岸鐓╅悗娑欏姉婢ф寧鎷呴幘鎵佸亾?)]
        public GameObject shaftObject;

#endif
#endif
        [Header("Collision Shape")]
        public float jawSelfRollAngle = 90f;
        public float collisionMargin = FourXCollisionMargin;
        public float capsuleRadius = FourXJawCapsuleRadius;
        public float shaftRadius = FourXShaftCapsuleRadius;
        [Range(3, 10)] public int jawCapsuleCount = 10;
        [Range(0.25f, 1.5f)] public float jawCapsuleFitRadiusScale = 0.75f;
        public bool showCapsuleGizmo = true;
        [Tooltip("Keeps the long shaft out of the soft-body contact broad phase. Enable only when shaft collision is required.")]
        public bool includeShaftCollision;
        [Range(-0.5f, 0.5f)] public float jawHorizontalCenterOffset = 0.012f;
        [Range(-0.5f, 0.5f)] public float jawVerticalCenterOffset = 0.012f;
        [Range(0f, 5f)] public float jawContactFrictionMultiplier = 1000f;

        [Header("Jaw Motion")]
        public float maxOpenAngle = 50f;
        public float openCloseSpeed = 60f;

        [Header("Grasping")]
        [Min(0f)] public float gaussianHeight = 0.02f;
        [Range(0.1f, 1f)] public float gaussianWidth = 0.5f;
        [Range(0.1f, 1f)] public float gaussianHardCoreRadius = 0.55f;
        [Min(0f)] public float gaussianFormDuration = 0.15f;
        [Min(1)] public int captureRetryPhysicsSteps = 5;

        [Header("Keyboard Control")]
        public float moveSpeed = 0.3f;
        public float rotateSpeed = 60f;
        [Tooltip("Multiplier applied while either Shift key is held.")]
        public float boostMultiplier = 3f;
        [Tooltip("Maximum keyboard translation per frame, expressed in capsule radii. The limit also scales while Shift boost is held so the boost remains effective without removing collision safety.")]
        [Min(0.01f)] public float maxMovementPerFrameRadiusFactor = 0.5f;

        [Header("Appearance")]
        public Color toolColor = new Color(0.7f, 0.7f, 0.78f, 1f);

        [Header("Scene Objects")]
        public GameObject upperJawObject;
        public GameObject lowerJawObject;
        public GameObject shaftObject;

        GameObject _jawUpObj, _jawDownObj, _shaftObj;
        bool _ownsJawUpObject, _ownsJawDownObject, _ownsShaftObject;
        Vector3[] _colVertsUp, _colVertsDown, _colVertsShaft;
        int[] _colTrisUp, _colTrisDown, _colTrisShaft;
        public bool IsGrasping { get; private set; }
        public float CurrentAngle => _currentAngle;
        public bool HasControlInput { get; private set; }
        public bool WantsToolContact => HasControlInput || _wantClose || IsGrasping;

        TetMeshData _data;
        XPBDSolverGPU _solver;
        TetMeshVisualizer _visualizer;
        Vector3 _toolPos;
        float _toolRotY;
        float _currentAngle;
        bool _wantClose;
        bool _initialized;

        Vector3 _pivotLocal;
        Vector3 _tipLocal;
        Vector3 _upperJawSelfAxisPointLocal;
        Vector3 _lowerJawSelfAxisPointLocal;
        const int MaxGripperCapsules = 12;

        struct LocalCapsule
        {
            public Vector3 a;
            public Vector3 b;
            public float radius;
        }

        readonly List<LocalCapsule> _upperJawLocalCapsules = new List<LocalCapsule>(5);
        readonly List<LocalCapsule> _lowerJawLocalCapsules = new List<LocalCapsule>(5);
        readonly Vector3[] _capsuleA = new Vector3[MaxGripperCapsules];
        readonly Vector3[] _capsuleB = new Vector3[MaxGripperCapsules];
        readonly Vector3[] _prevCapsuleA = new Vector3[MaxGripperCapsules];
        readonly Vector3[] _prevCapsuleB = new Vector3[MaxGripperCapsules];
        readonly float[] _capsuleR = new float[MaxGripperCapsules];
        readonly float[] _capsuleFriction = new float[MaxGripperCapsules];
        readonly Vector3[] _lastCapsuleA = new Vector3[MaxGripperCapsules];
        readonly Vector3[] _lastCapsuleB = new Vector3[MaxGripperCapsules];
        readonly Vector3[] _dbgCapsuleA = new Vector3[MaxGripperCapsules];
        readonly Vector3[] _dbgCapsuleB = new Vector3[MaxGripperCapsules];
        readonly float[] _dbgCapsuleR = new float[MaxGripperCapsules];
        int _lastCapsuleCount;
        int _dbgCapsuleCount;
        int _dbgUpperCapsuleCount;
        int _dbgLowerCapsuleCount;
        Vector3 _dbgBBoxMin;
        Vector3 _dbgBBoxMax;
        bool _hasPrevCapsules;

        struct GraspedParticle
        {
            public int index;
            public float originalInvMass;
            public Vector3 frameCoordinates;
            public float gaussianWeight;
        }

        readonly List<GraspedParticle> _graspedParticles = new List<GraspedParticle>();
        readonly List<int> _graspedParticleIndices = new List<int>();
        Vector3 _graspCenterToolLocal;
        float _graspShapeBlend;
        int _transitionParticleCount;
        int _captureAttemptCount;
        bool _captureGaveUp;
 #if false
        bool _ownsJawUpObject, _ownsJawDownObject, _ownsShaftObject;

        // 缁炬壆澧楅幐鎺旂磾閹寸偟澹愰柡浣哄瀹?(闁哄牜鍓欏﹢鎾锤閹邦厾鍨奸柨娑樿嫰閸戯紕绱撻埡鍌涙澒)
#if false
        Vector3[] _colVertsUp, _colVertsDown, _colVertsShaft;
        int[]     _colTrisUp,  _colTrisDown,  _colTrisShaft;

        // 闂佸墽澧楃敮鎾矗閸屾稒娈?(濞寸姴瀛╄啯闁搞劌顑嗙敮褰掑棘?
        // 濞戞挸锕ｇ粭鍛紣濮樿京鎼?Z 閺夌偛鐡ㄩ弻鐔煎触閹寸姵鐣遍梺鍓у鐢挳鎮欑憴鍕棆閺?        // 闂佸墽澧楃敮鎾倷閻熻埇浜ｇ紒鎾呯畱濠€?Y=9.0, Z=-25.5 (OBJ 闁秆勫姈閻?
        Vector3 _pivotLocal; // 缂傚倵鏅滈弬渚€宕ユ惔锝嗙暠闂佸墽澧楃敮鎾倷?(闁哄牜鍓欏﹢鎾锤閹邦厾鍨?
        Vector3 _tipLocal;   // 缂傚倵鏅滈弬渚€宕ユ惔锝嗙暠閻忓繑鐗滈?(闁哄牜鍓欏﹢鎾锤閹邦厾鍨?
        Vector3 _upperJawSelfAxisPointLocal;
        Vector3 _lowerJawSelfAxisPointLocal;

        // 闁煎疇娉涘▔顓熸媴閹捐尙鐟柣锝呰嫰濞兼寮?(闁活潿鍔嬬花?Gizmo 闁告瑯鍨甸～瀣礌?
        const int MaxGripperCapsules = 12;

        struct LocalCapsule
        {
            public Vector3 a, b;
            public float radius;
        }

        readonly List<LocalCapsule> _upperJawLocalCapsules = new List<LocalCapsule>(5);
        readonly List<LocalCapsule> _lowerJawLocalCapsules = new List<LocalCapsule>(5);
        readonly Vector3[] _capsuleA = new Vector3[MaxGripperCapsules];
        readonly Vector3[] _capsuleB = new Vector3[MaxGripperCapsules];
        readonly Vector3[] _prevCapsuleA = new Vector3[MaxGripperCapsules];
        readonly Vector3[] _prevCapsuleB = new Vector3[MaxGripperCapsules];
        readonly float[] _capsuleR = new float[MaxGripperCapsules];
        readonly float[] _capsuleFriction = new float[MaxGripperCapsules];
        readonly Vector3[] _lastCapsuleA = new Vector3[MaxGripperCapsules];
        readonly Vector3[] _lastCapsuleB = new Vector3[MaxGripperCapsules];
        readonly Vector3[] _dbgCapsuleA = new Vector3[MaxGripperCapsules];
        readonly Vector3[] _dbgCapsuleB = new Vector3[MaxGripperCapsules];
        readonly float[] _dbgCapsuleR = new float[MaxGripperCapsules];
        int _lastCapsuleCount;
        int _dbgCapsuleCount;
        int _dbgUpperCapsuleCount;
        int _dbgLowerCapsuleCount;
        Vector3 _dbgBBoxMin, _dbgBBoxMax;
        bool _hasPrevCapsules;

        // Captured particles retain their original inverse mass for release.
        struct GraspedParticle
        {
            public int index;
            public float originalInvMass;
            public Vector3 frameCoordinates; // 闁硅埖鎸歌ぐ鍥锤閹邦厾鍨肩紒顖濐唺閼垫垿鎯?(u, v, 闁告ɑ鑹剧€?
            public float gaussianWeight;
        }
        readonly List<GraspedParticle> _graspedParticles = new List<GraspedParticle>();
        readonly List<int> _graspedParticleIndices = new List<int>();
        Vector3 _graspCenterToolLocal;
        float _graspShapeBlend;
        int _transitionParticleCount;
        int _captureAttemptCount;
        bool _captureGaveUp;

        // 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺?
        // 闁告帗绻傞～鎰板礌?        // 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺?
#endif
#endif
        public void Init(TetMeshData data, XPBDSolverGPU solver, TetMeshVisualizer visualizer)
        {
            _data = data;
            _solver = solver;
            _visualizer = visualizer;

            ApplyGripperScalePreset();

            _toolPos = transform.position;
            _toolRotY = 0f;
            _currentAngle = maxOpenAngle; // 闁告帗绻傞～鎰嚕閻樿尙纾?
            _wantClose = false;
            IsGrasping = false;
            _graspedParticles.Clear();
            _graspedParticleIndices.Clear();
            _graspShapeBlend = 0f;
            _transitionParticleCount = 0;
            ResetCaptureRetryState();
            _uploadFrame = 0; // 闂佹彃绉堕悿鍡欐嫚婵犲啯鐒介悹浣插墲閺嗙喖宕?
            // 闂佸墽澧楃敮鎾倷?(OBJ 闁秆勫姈閻?闁?缂傚倵鏅滈弬渚€宕?
            _pivotLocal = new Vector3(0f, 9.0f, -25.5f) * modelScale;
            // 濠㈣泛婀遍崺鍛焊閺嶎偒浼?(OBJ 闁秆勫姈閻?闁?缂傚倵鏅滈弬渚€宕?
            _tipLocal = new Vector3(0f, 9.0f, -43.0f) * modelScale;
            // 闁哄妫滈棅鈺冧焊閸撗屼紓 (闁归潧顑嗛悞娲棘閻熺増鍊婚柨娑樿嫰濞嗐垺瀵煎灞藉枙濠㈠墎鍠栭弳杈ㄧ閵夘煈娲柣鈺傜墬閺嗭絾绋夐鍛秳闂?
            _shaftEndLocal = new Vector3(0f, 9.0f, 307.0f) * modelScale;
            _shaftTipLocal = _pivotLocal;
            _prevToolPos = _toolPos;
            _hasPrevCapsules = false;

            LoadModels();
            LoadCollisionMeshes();
            _initialized = true;

            Debug.Log($"[GripperTool] Init | scale={modelScale} | jawCapsules={Mathf.Clamp(jawCapsuleCount, 3, 10)} + shaft");
        }

        void ApplyGripperScalePreset()
        {
            if (!useFourXGripperPreset) return;

            modelScale = FourXModelScale;
            collisionMargin = FourXCollisionMargin;
            capsuleRadius = FourXJawCapsuleRadius;
            shaftRadius = FourXShaftCapsuleRadius;
        }

        // 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺?
        // Update 闁?闂佹鍠氬ú蹇旀綇閹惧啿寮?+ 闁告瑯鍨甸～瀣礌?        // 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺?
        void Update()
        {
            if (!_initialized) return;

            // 闁冲厜鍋撻柍鍏夊亾 缂佸顕ф慨鈺呮晬閸︽オHVBN闁挎稑顦弨銏ゅ煘閳ь剟鍩為埀顒勫煘閳ь剟鍩為埀顒勫煘閳ь剟鍩為埀顒勫煘閳ь剟鍩為埀顒勫煘閳ь剟鍩為埀顒勫煘閳ь剟鍩為埀顒勫煘閳ь剟鍩為埀顒勫煘閳ь剟鍩為埀顒勫煘閳ь剟鍩為埀顒勫煘閳ь剟鍩為埀顒勫煘閳ь剟鍩為埀顒勫煘閳ь剟鍩為埀顒勫煘閳ь剟鍩為埀顒勫煘閳ь剟鍩為埀顒勫煘閳?
            Vector3 move = Vector3.zero;
            if (Input.GetKey(KeyCode.H)) move.x += 1f;
            if (Input.GetKey(KeyCode.F)) move.x -= 1f;
            if (Input.GetKey(KeyCode.G)) move.y += 1f;
            if (Input.GetKey(KeyCode.B)) move.y -= 1f;
            if (Input.GetKey(KeyCode.N)) move.z += 1f;
            if (Input.GetKey(KeyCode.V)) move.z -= 1f;

            bool rotateLeft = Input.GetKey(KeyCode.Keypad1);
            bool rotateRight = Input.GetKey(KeyCode.Keypad3);
            bool closeHeld = Input.GetKey(KeyCode.Z);
            bool openHeld = Input.GetKey(KeyCode.X);
            HasControlInput = move.sqrMagnitude > 1e-8f || rotateLeft || rotateRight || closeHeld || openHeld;

            bool boost = Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift);
            float speed = boost ? moveSpeed * boostMultiplier : moveSpeed;
            if (move.sqrMagnitude > 1e-8f)
            {
                Vector3 desiredMove = move.normalized * speed * Time.deltaTime;
                // 闁?闂侇偆鍠庣€规娊鏌﹂幒鎴濈厬: 婵絽绻愰幎姘跺嫉閳ь剚寰勮浜涢柛?capsuleRadius * 0.5闁挎稑鐭傚Σ璇差潰閵忋倖鐦堢紒?                // 闁告瑥鍊介埀?SOFA alarmDistance 婵帒鍊告惔?
                float maxMove = capsuleRadius * maxMovementPerFrameRadiusFactor *
                                (boost ? Mathf.Max(1f, boostMultiplier) : 1f);
                if (desiredMove.magnitude > maxMove)
                    desiredMove = desiredMove.normalized * maxMove;
                _toolPos += desiredMove;
            }

            // 闁冲厜鍋撻柍鍏夊亾 闁哄啫顑堝ù鍡涙晬閸︻暃mpad 1/3闁挎稑顦弨銏ゅ煘閳ь剟鍩為埀顒勫煘閳ь剟鍩為埀顒勫煘閳ь剟鍩為埀顒勫煘閳ь剟鍩為埀顒勫煘閳ь剟鍩為埀顒勫煘閳ь剟鍩為埀顒勫煘閳ь剟鍩為埀顒勫煘閳ь剟鍩為埀顒勫煘閳ь剟鍩為埀顒勫煘閳ь剟鍩為埀顒勫煘閳ь剟鍩為埀顒勫煘閳ь剟鍩為埀顒勫煘閳?
            if (rotateLeft) _toolRotY -= rotateSpeed * Time.deltaTime;
            if (rotateRight) _toolRotY += rotateSpeed * Time.deltaTime;

            // Z/X continuous jaw control: hold to move, release to stop.
            float jawStep = openCloseSpeed * Time.deltaTime;
            if (closeHeld && !openHeld)
            {
                _wantClose = true;
                _currentAngle = Mathf.MoveTowards(_currentAngle, 0f, jawStep);
            }
            else if (openHeld && !closeHeld)
            {
                _wantClose = false;
                _currentAngle = Mathf.MoveTowards(_currentAngle, maxOpenAngle, jawStep);
            }

            UpdateVisual();
        }

        // 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺?
        // PhysicsStep 闁?闁?FixedUpdate 濞戞搩鍙€椤?SoftBody 閻犲鍟伴弫?
        // 缁炬壆澧楅幐鎺戭啅閼碱剚鏆?GPU CSToolCollision 濠㈣泛瀚幃濠囨晬瀹€鍐闂佹彃鑻ぐ褔宕戝顑句粴闁?        // 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺?
        public void PhysicsStep()
        {
            if (!_initialized || _data == null || _solver == null) return;

            Transform tf = _visualizer != null ? _visualizer.transform : transform;

            // Contact is solved on the GPU; this code owns the capture state.
            bool isFullyClosed = _wantClose && _currentAngle <= 25f;
            if (!isFullyClosed && !IsGrasping)
                ResetCaptureRetryState();

            if (isFullyClosed && !IsGrasping)
            {
                TryCaptureParticles(tf);
            }
            else if (IsGrasping && isFullyClosed)
            {
                UpdateGraspedParticles(tf);
            }
            else if (IsGrasping && !_wantClose)
            {
                ReleaseParticles();
                IsGrasping = false;
                ResetCaptureRetryState();
            }
        }

        void TryCaptureParticles(Transform meshTf)
        {
            if (_captureGaveUp)
                return;

            int maxAttempts = Mathf.Max(1, captureRetryPhysicsSteps);
            _captureAttemptCount++;
            IsGrasping = CaptureParticles(meshTf);

            if (IsGrasping)
                return;

            if (_captureAttemptCount >= maxAttempts)
            {
                _captureGaveUp = true;
                Debug.Log($"[GripperTool] Capture gave up after {_captureAttemptCount} physics steps.");
            }
        }

        void ResetCaptureRetryState()
        {
            _captureAttemptCount = 0;
            _captureGaveUp = false;
        }

        // 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺?
        // GPU 缁炬壆澧楅幐? 濠㈣泛婀遍崺?3~10 濞戞搩浜濈€氭瑩宕ラ崼锝呭帯闁搞儱锕ｇ紞?+ 1 濞戞搩浜濆宀勭叕椤愩儱鍘￠柛銉ワ梗缂?
        // 闁告瑥鍊介埀?SOFA: proximity/contact pipeline + CCD
        // 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺?
        int _uploadFrame = 0;
        Vector3 _prevToolPos;
        Vector3 _shaftEndLocal; // 闁哄妫滈棅鈺冧焊閸撗屼紓(闁归潧顑嗛悞娲棘閻熺増鍊?

        Vector3 _shaftTipLocal; // shaft tip/hinge end in scaled OBJ local space

        public void UploadToolCollisionToGPU()
        {
            if (!_initialized || _solver == null) return;

            Quaternion toolRot = Quaternion.Euler(0, _toolRotY, 0);
            Quaternion upperRot = Quaternion.AngleAxis(_currentAngle, Vector3.right);
            Quaternion lowerRot = Quaternion.AngleAxis(-_currentAngle, Vector3.right);

            int capsuleCount = 0;
            int upperAdded = AddJawCapsules(
                _upperJawLocalCapsules, true, upperRot, toolRot, ref capsuleCount);
            int lowerAdded = AddJawCapsules(
                _lowerJawLocalCapsules, false, lowerRot, toolRot, ref capsuleCount);

            if (includeShaftCollision)
            {
                Vector3 shaftA = toolRot * _shaftEndLocal + _toolPos;
                Vector3 shaftB = toolRot * _shaftTipLocal + _toolPos;
                AddCapsule(shaftA, shaftB, shaftRadius, ref capsuleCount);
            }

            for (int i = 0; i < MaxGripperCapsules; i++)
                _capsuleFriction[i] = 0f;

            int jawCapsuleCount = upperAdded + lowerAdded;
            float jawFriction = IsGrasping ? 0f : Mathf.Max(0f, jawContactFrictionMultiplier);
            for (int i = 0; i < jawCapsuleCount && i < capsuleCount; i++)
                _capsuleFriction[i] = jawFriction;

            for (int i = 0; i < capsuleCount; i++)
            {
                if (_hasPrevCapsules && i < _lastCapsuleCount)
                {
                    _prevCapsuleA[i] = _lastCapsuleA[i];
                    _prevCapsuleB[i] = _lastCapsuleB[i];
                }
                else
                {
                    _prevCapsuleA[i] = _capsuleA[i];
                    _prevCapsuleB[i] = _capsuleB[i];
                }
            }

            _solver.SetCapsuleCollisionParams(
                _capsuleA,
                _capsuleB,
                _capsuleR,
                _prevCapsuleA,
                _prevCapsuleB,
                capsuleCount,
                _capsuleFriction);

            _dbgCapsuleCount = capsuleCount;
            _dbgUpperCapsuleCount = upperAdded;
            _dbgLowerCapsuleCount = lowerAdded;
            _dbgBBoxMin = Vector3.positiveInfinity;
            _dbgBBoxMax = Vector3.negativeInfinity;
            for (int i = 0; i < capsuleCount; i++)
            {
                _dbgCapsuleA[i] = _capsuleA[i];
                _dbgCapsuleB[i] = _capsuleB[i];
                _dbgCapsuleR[i] = _capsuleR[i];

                float margin = collisionMargin + _capsuleR[i];
                Vector3 pad = Vector3.one * margin;
                _dbgBBoxMin = Vector3.Min(_dbgBBoxMin, Vector3.Min(_capsuleA[i], _capsuleB[i]) - pad);
                _dbgBBoxMax = Vector3.Max(_dbgBBoxMax, Vector3.Max(_capsuleA[i], _capsuleB[i]) + pad);
            }

            if (capsuleCount == 0)
            {
                _dbgBBoxMin = Vector3.zero;
                _dbgBBoxMax = Vector3.zero;
            }

            if (_uploadFrame < 3)
            {
                Debug.Log($"[GripperTool] capsule collision F{_uploadFrame}: " +
                    $"jawCaps={upperAdded + lowerAdded} upper={upperAdded} lower={lowerAdded} " +
                    $"total={capsuleCount} shaftR={shaftRadius:F4}");
                _uploadFrame++;
            }

            for (int i = 0; i < capsuleCount; i++)
            {
                _lastCapsuleA[i] = _capsuleA[i];
                _lastCapsuleB[i] = _capsuleB[i];
            }
            _lastCapsuleCount = capsuleCount;
            _hasPrevCapsules = true;
            _prevToolPos = _toolPos;
        }

        int AddJawCapsules(
            List<LocalCapsule> localCapsules,
            bool isUpperJaw,
            Quaternion jawRot,
            Quaternion toolRot,
            ref int capsuleCount)
        {
            if (localCapsules == null || localCapsules.Count == 0)
                return 0;

            int added = 0;
            for (int i = 0; i < localCapsules.Count; i++)
            {
                LocalCapsule c = localCapsules[i];
                Vector3 a = TransformJawLocalPoint(c.a, isUpperJaw, jawRot, toolRot);
                Vector3 b = TransformJawLocalPoint(c.b, isUpperJaw, jawRot, toolRot);
                if (AddCapsule(a, b, c.radius, ref capsuleCount))
                    added++;
            }
            return added;
        }

        Vector3 TransformJawLocalPoint(
            Vector3 localPoint,
            bool isUpperJaw,
            Quaternion jawRot,
            Quaternion toolRot)
        {
            Vector3 selfRolledPoint = RotateAroundJawSelfAxis(localPoint, isUpperJaw);
            Vector3 v = selfRolledPoint - _pivotLocal;
            v = jawRot * v;
            v = Quaternion.AngleAxis(jawRollAngle, Vector3.forward) * v;
            v += _pivotLocal;
            v += GetJawCenteringOffsetLocal(isUpperJaw);
            return toolRot * v + _toolPos;
        }

        Vector3 GetJawCenteringOffsetLocal(bool isUpperJaw)
        {
            float sign = isUpperJaw ? -1f : 1f;
            return new Vector3(
                sign * jawHorizontalCenterOffset,
                sign * jawVerticalCenterOffset,
                0f);
        }

        Vector3 RotateAroundJawSelfAxis(Vector3 point, bool isUpperJaw)
        {
            Vector3 axisPoint = isUpperJaw
                ? _upperJawSelfAxisPointLocal
                : _lowerJawSelfAxisPointLocal;
            Quaternion selfRoll = Quaternion.AngleAxis(jawSelfRollAngle, Vector3.forward);
            return axisPoint + selfRoll * (point - axisPoint);
        }

        bool AddCapsule(Vector3 a, Vector3 b, float radius, ref int capsuleCount)
        {
            if (capsuleCount >= MaxGripperCapsules)
                return false;

            _capsuleA[capsuleCount] = a;
            _capsuleB[capsuleCount] = b;
            _capsuleR[capsuleCount] = Mathf.Max(0.0001f, radius);
            capsuleCount++;
            return true;
        }


        // 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺?
        // Phase 2: 濠㈠墎鎳撹ぐ?
        // 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺?
        bool CaptureParticles(Transform meshTf)
        {
            _graspedParticles.Clear();
            _graspedParticleIndices.Clear();
            _graspShapeBlend = 0f;
            _transitionParticleCount = 0;

            // 濠㈣泛婀遍崺鍛焊閺嶎偒浼傞柛鏍ф惈閻?(OBJ Z 闁?-43 闁?濞戞挻鐗滈弲顐﹀锤閹邦厾鍨?
            // 濞戞挸锕ｇ粭鍛紣濮樿埖锛旈柛姘墛濡炲倿鏁嶇仦濮愪粴濞达絽绻愰悾鐘崇椤戣法顓洪梻鍌氼嚟濞堟垹鍒掗幒鎴犳憤
            var worldVertsUp   = TransformCollisionVerts(_colVertsUp, true);
            var worldVertsDown = TransformCollisionVerts(_colVertsDown, false);

            // 闁活潿鍔嬬悮杈╃磼閸曨収娼鹃柟鍓у仱閵嗗﹪鎮欓崷顓熺暠 AABB 濞存嚎鍊濆▔锔芥媴濠娾偓鐠愮喐寰勯悷鏉跨悼闁告牕鎼悡?
            Vector3 bboxMin = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
            Vector3 bboxMax = new Vector3(float.MinValue, float.MinValue, float.MinValue);
            foreach (var v in worldVertsUp)
            { bboxMin = Vector3.Min(bboxMin, v); bboxMax = Vector3.Max(bboxMax, v); }
            foreach (var v in worldVertsDown)
            { bboxMin = Vector3.Min(bboxMin, v); bboxMax = Vector3.Max(bboxMax, v); }

            // 缂傚倵鏅涢惃顒佺▔閳ь剟鎮?bbox 闁告瑯浜滆ぐ鍥冀缁嬭法濡囬柛鏍ф惈閻?
            Vector3 shrink = (bboxMax - bboxMin) * 0.1f;
            bboxMin += shrink;
            bboxMax -= shrink;

            // 閺夌儐鍓氬畷鏌ュ礆?mesh local space
            Vector3 localMin = meshTf.InverseTransformPoint(bboxMin);
            Vector3 localMax = meshTf.InverseTransformPoint(bboxMax);
            // 缁绢収鍠曠换?min < max
            Vector3 realMin = Vector3.Min(localMin, localMax);
            Vector3 realMax = Vector3.Max(localMin, localMax);

            Vector3 jawCenterWorld = (bboxMin + bboxMax) * 0.5f;
            Quaternion toolRot = Quaternion.Euler(0f, _toolRotY, 0f);
            _graspCenterToolLocal = Quaternion.Inverse(toolRot) * (jawCenterWorld - _toolPos);
            GetGraspFrame(out Vector3 frameCenter, out Vector3 axisU,
                out Vector3 axisV, out Vector3 towardCrotch);

            var candidateIndices = new List<int>();
            var candidateCoordinates = new List<Vector3>();
            float halfU = 0f;
            float halfV = 0f;
            for (int i = 0; i < _data.NumParticles; i++)
            {
                if (_data.InvMass[i] == 0f) continue;
                Vector3 p = _data.Positions[i];
                if (p.x >= realMin.x && p.x <= realMax.x &&
                    p.y >= realMin.y && p.y <= realMax.y &&
                    p.z >= realMin.z && p.z <= realMax.z)
                {
                    Vector3 worldOffset = meshTf.TransformPoint(p) - frameCenter;
                    Vector3 coordinates = new Vector3(
                        Vector3.Dot(worldOffset, axisU),
                        Vector3.Dot(worldOffset, axisV),
                        Vector3.Dot(worldOffset, towardCrotch));
                    candidateIndices.Add(i);
                    candidateCoordinates.Add(coordinates);
                    halfU = Mathf.Max(halfU, Mathf.Abs(coordinates.x));
                    halfV = Mathf.Max(halfV, Mathf.Abs(coordinates.y));
                }
            }

            if (candidateIndices.Count == 0)
            {
                Debug.Log("[Stage2GripperTool] No particles found between the jaws.");
                return false;
            }

            halfU = Mathf.Max(halfU, 0.0001f);
            halfV = Mathf.Max(halfV, 0.0001f);
            float coreRadius = Mathf.Clamp(gaussianHardCoreRadius, 0.1f, 1f);
            float sigmaU = Mathf.Max(halfU * Mathf.Clamp(gaussianWidth, 0.1f, 1f), 0.0001f);
            float sigmaV = Mathf.Max(halfV * Mathf.Clamp(gaussianWidth, 0.1f, 1f), 0.0001f);
            int nearestCandidate = -1;
            float nearestRadius = float.MaxValue;

            for (int c = 0; c < candidateIndices.Count; c++)
            {
                Vector3 coordinates = candidateCoordinates[c];
                float normalizedU = coordinates.x / halfU;
                float normalizedV = coordinates.y / halfV;
                float normalizedRadius = Mathf.Sqrt(
                    normalizedU * normalizedU + normalizedV * normalizedV);

                if (normalizedRadius < nearestRadius)
                {
                    nearestRadius = normalizedRadius;
                    nearestCandidate = c;
                }

                if (normalizedRadius > coreRadius)
                {
                    _transitionParticleCount++;
                    continue;
                }

                float gaussianWeight = EvaluateGaussianWeight(coordinates, sigmaU, sigmaV);
                LockGraspParticle(candidateIndices[c], coordinates, gaussianWeight);
            }

            // Keep the nearest candidate when the Gaussian core is smaller than the particle spacing.
            if (_graspedParticles.Count == 0 && nearestCandidate >= 0)
            {
                Vector3 coordinates = candidateCoordinates[nearestCandidate];
                float gaussianWeight = EvaluateGaussianWeight(coordinates, sigmaU, sigmaV);
                LockGraspParticle(candidateIndices[nearestCandidate], coordinates, gaussianWeight);
                _transitionParticleCount = Mathf.Max(0, _transitionParticleCount - 1);
            }

            _solver.UploadInvMass(_data);
            Debug.Log($"[Stage2GripperTool] Captured core={_graspedParticles.Count}, transition={_transitionParticleCount}.");
            return _graspedParticles.Count > 0;
        }

        void UpdateGraspedParticles(Transform meshTf)
        {
            if (_graspedParticles.Count == 0) return;

            if (gaussianFormDuration <= 0f)
                _graspShapeBlend = 1f;
            else
                _graspShapeBlend = Mathf.MoveTowards(
                    _graspShapeBlend, 1f, Time.fixedDeltaTime / gaussianFormDuration);

            float smoothBlend = Mathf.SmoothStep(0f, 1f, _graspShapeBlend);
            GetGraspFrame(out Vector3 frameCenter, out Vector3 axisU,
                out Vector3 axisV, out Vector3 towardCrotch);

            foreach (var gp in _graspedParticles)
            {
                Vector3 coordinates = gp.frameCoordinates;
                float gaussianOffset = Mathf.Max(0f, gaussianHeight) * gp.gaussianWeight * smoothBlend;
                Vector3 targetWorld = frameCenter +
                    axisU * coordinates.x +
                    axisV * coordinates.y +
                    towardCrotch * (coordinates.z + gaussianOffset);
                Vector3 targetLocal = meshTf.InverseTransformPoint(targetWorld);

                _data.Positions[gp.index] = targetLocal;
                _data.PrevPositions[gp.index] = targetLocal;
                _data.Velocities[gp.index] = Vector3.zero;
            }
            _solver.UploadParticleStates(_data, _graspedParticleIndices);
        }

        void ReleaseParticles()
        {
            foreach (var gp in _graspedParticles)
            {
                _data.InvMass[gp.index] = gp.originalInvMass;
                _data.Velocities[gp.index] *= 0.1f;
            }
            int n = _graspedParticles.Count;
            _graspedParticles.Clear();
            _graspedParticleIndices.Clear();
            _graspShapeBlend = 0f;
            _transitionParticleCount = 0;
            _solver.UploadInvMass(_data);
            Debug.Log($"[Stage2GripperTool] Released {n} particles.");
        }

        void LockGraspParticle(int index, Vector3 frameCoordinates, float gaussianWeight)
        {
            _graspedParticles.Add(new GraspedParticle
            {
                index = index,
                originalInvMass = _data.InvMass[index],
                frameCoordinates = frameCoordinates,
                gaussianWeight = gaussianWeight
            });
            _graspedParticleIndices.Add(index);
            _data.InvMass[index] = 0f;
        }

        static float EvaluateGaussianWeight(Vector3 coordinates, float sigmaU, float sigmaV)
        {
            float u = coordinates.x / sigmaU;
            float v = coordinates.y / sigmaV;
            return Mathf.Exp(-0.5f * (u * u + v * v));
        }

        void GetGraspFrame(out Vector3 center, out Vector3 axisU,
            out Vector3 axisV, out Vector3 towardCrotch)
        {
            Quaternion toolRot = Quaternion.Euler(0f, _toolRotY, 0f);
            Quaternion jawMountRot = Quaternion.AngleAxis(jawRollAngle, Vector3.forward);
            center = toolRot * _graspCenterToolLocal + _toolPos;
            towardCrotch = (toolRot * (_pivotLocal - _tipLocal)).normalized;

            axisU = toolRot * (jawMountRot * Vector3.right);
            axisU = Vector3.ProjectOnPlane(axisU, towardCrotch).normalized;
            if (axisU.sqrMagnitude < 1e-8f)
                axisU = Vector3.ProjectOnPlane(toolRot * Vector3.up, towardCrotch).normalized;
            axisV = Vector3.Cross(towardCrotch, axisU).normalized;
        }

        // 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺?
        // 闁秆勫姈閻栵綁宕ｅΟ璇插簥
        // 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺?
        /// <summary>閻忓繐妫涢～顐﹀箻閻愮數绉归柡宥呫偢閵嗗﹪鎮欓悷鏉跨秮闁瑰箍鍨归崺灞剧▔閺嶎偅娅曢柛褎鍔栭悥锝夋晬閸繃鍎撶€殿喒鍋撻柛姘墛濡棙娼濠勭</summary>
        Vector3[] TransformCollisionVerts(Vector3[] localVerts, bool isUpperJaw)
        {
            var result = new Vector3[localVerts.Length];
            float jawAngle = isUpperJaw ? _currentAngle : -_currentAngle;
            Quaternion jawRot = Quaternion.AngleAxis(jawAngle, Vector3.right);
            Quaternion jawMountRot = Quaternion.AngleAxis(jawRollAngle, Vector3.forward);
            Quaternion toolRot = Quaternion.Euler(0, _toolRotY, 0);

            for (int i = 0; i < localVerts.Length; i++)
            {
                Vector3 selfRolledPoint = RotateAroundJawSelfAxis(localVerts[i], isUpperJaw);
                Vector3 v = selfRolledPoint - _pivotLocal;
                v = jawRot * v;
                v = jawMountRot * v;
                v += _pivotLocal;
                v += GetJawCenteringOffsetLocal(isUpperJaw);
                result[i] = toolRot * v + _toolPos;
            }
            return result;
        }
        Vector3 GetJawTipCenter()
        {
            Quaternion toolRot = Quaternion.Euler(0, _toolRotY, 0);
            return toolRot * _tipLocal + _toolPos;
        }

        /// <summary>CPU 缂?Point-to-Capsule 閻犵儤绻勯‖鍥晬閸垺鏆忓ù婊冩唉閻︽牠寮銊х</summary>
        static float PointCapsuleDist(Vector3 p, Vector3 a, Vector3 b)
        {
            Vector3 ab = b - a;
            float ab2 = Vector3.Dot(ab, ab);
            float t = ab2 < 1e-10f ? 0f : Mathf.Clamp01(Vector3.Dot(p - a, ab) / ab2);
            Vector3 closest = a + t * ab;
            return (p - closest).magnitude;
        }

        // 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺?
        // OBJ 闁告梻濮惧ù?
        // 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺?
        void LoadModels()
        {
            var mat = new Material(
                Shader.Find("Universal Render Pipeline/Lit") ??
                Shader.Find("Standard"));
            mat.color = toolColor;

            _jawUpObj = CreateOrReuseMeshObject(
                upperJawObject, "Upper Jaw", jawUpVisual, mat, out _ownsJawUpObject);
            _jawDownObj = CreateOrReuseMeshObject(
                lowerJawObject, "Lower Jaw", jawDownVisual, mat, out _ownsJawDownObject);
            _shaftObj = CreateOrReuseMeshObject(
                shaftObject, "Shaft", shaftVisual, mat, out _ownsShaftObject);
        }

        void LoadCollisionMeshes()
        {
            string dir = Application.streamingAssetsPath;

            LoadOBJData(Path.Combine(dir, jawUpCol),   out _colVertsUp,   out _colTrisUp);
            LoadOBJData(Path.Combine(dir, jawDownCol), out _colVertsDown, out _colTrisDown);
            LoadOBJData(Path.Combine(dir, shaftCol),   out _colVertsShaft, out _colTrisShaft);

            if (_colVertsUp != null)
                for (int i = 0; i < _colVertsUp.Length; i++)
                    _colVertsUp[i] *= modelScale;
            if (_colVertsDown != null)
                for (int i = 0; i < _colVertsDown.Length; i++)
                    _colVertsDown[i] *= modelScale;
            if (_colVertsShaft != null)
                for (int i = 0; i < _colVertsShaft.Length; i++)
                    _colVertsShaft[i] *= modelScale;

            _upperJawSelfAxisPointLocal = CalculateJawSelfAxisPoint(_colVertsUp);
            _lowerJawSelfAxisPointLocal = CalculateJawSelfAxisPoint(_colVertsDown);

            Debug.Log($"[GripperTool] 缁炬壆澧楅幐鎺旂磾閹寸偟澹? Up {_colVertsUp?.Length}V/{_colTrisUp?.Length/3}F " +
                      $"| Down {_colVertsDown?.Length}V/{_colTrisDown?.Length/3}F " +
                      $"| Shaft {_colVertsShaft?.Length}V/{_colTrisShaft?.Length/3}F");

            FitShaftCapsuleFromCollisionMesh();
            FitJawCapsules(_colVertsUp, _upperJawLocalCapsules, GetUpperJawCapsuleTargetCount(), "UpperJaw");
            FitJawCapsules(_colVertsDown, _lowerJawLocalCapsules, GetLowerJawCapsuleTargetCount(), "LowerJaw");
        }

        Vector3 CalculateJawSelfAxisPoint(Vector3[] verts)
        {
            if (verts == null || verts.Length == 0)
                return _pivotLocal;

            Vector2 center = Vector2.zero;
            int count = 0;
            float workingEndZ = Mathf.Max(_pivotLocal.z, _tipLocal.z);
            for (int i = 0; i < verts.Length; i++)
            {
                if (verts[i].z > workingEndZ)
                    continue;

                center += new Vector2(verts[i].x, verts[i].y);
                count++;
            }

            if (count == 0)
                return _pivotLocal;

            center /= count;
            return new Vector3(center.x, center.y, _pivotLocal.z);
        }

        int GetUpperJawCapsuleTargetCount()
        {
            int total = Mathf.Clamp(jawCapsuleCount, 3, 10);
            return Mathf.Clamp((total + 1) / 2, 1, 5);
        }

        int GetLowerJawCapsuleTargetCount()
        {
            int total = Mathf.Clamp(jawCapsuleCount, 3, 10);
            return Mathf.Clamp(total / 2, 1, 5);
        }

        void FitJawCapsules(Vector3[] verts, List<LocalCapsule> result, int count, string label)
        {
            result.Clear();
            count = Mathf.Clamp(count, 1, 5);

            if (verts == null || verts.Length == 0)
            {
                FitFallbackJawCapsules(result, count);
                Debug.LogWarning($"[GripperTool] {label} collision mesh missing; using fallback jaw capsules.");
                return;
            }

            Vector3 min = verts[0];
            Vector3 max = verts[0];
            for (int i = 1; i < verts.Length; i++)
            {
                min = Vector3.Min(min, verts[i]);
                max = Vector3.Max(max, verts[i]);
            }

            float zMin = min.z;
            float zMax = max.z;
            float zSpan = Mathf.Max(0.0001f, zMax - zMin);
            float zOverlap = Mathf.Max(0.0001f, zSpan * 0.025f);
            float minRadius = Mathf.Max(0.0005f, capsuleRadius * 0.35f);
            float radiusScale = Mathf.Clamp(
                jawCapsuleFitRadiusScale <= 0f ? 1f : jawCapsuleFitRadiusScale,
                0.25f,
                1.5f);

            for (int c = 0; c < count; c++)
            {
                float z0 = Mathf.Lerp(zMin, zMax, c / (float)count);
                float z1 = Mathf.Lerp(zMin, zMax, (c + 1) / (float)count);
                Vector3 segMin = Vector3.positiveInfinity;
                Vector3 segMax = Vector3.negativeInfinity;
                int segVerts = 0;

                for (int i = 0; i < verts.Length; i++)
                {
                    Vector3 v = verts[i];
                    if (v.z < z0 - zOverlap || v.z > z1 + zOverlap)
                        continue;

                    segMin = Vector3.Min(segMin, v);
                    segMax = Vector3.Max(segMax, v);
                    segVerts++;
                }

                if (segVerts == 0)
                {
                    segMin = min;
                    segMax = max;
                }

                float centerX = (segMin.x + segMax.x) * 0.5f;
                float centerY = (segMin.y + segMax.y) * 0.5f;
                float radius = 0f;

                for (int i = 0; i < verts.Length; i++)
                {
                    Vector3 v = verts[i];
                    if (segVerts > 0 && (v.z < z0 - zOverlap || v.z > z1 + zOverlap))
                        continue;

                    float dx = v.x - centerX;
                    float dy = v.y - centerY;
                    radius = Mathf.Max(radius, Mathf.Sqrt(dx * dx + dy * dy));
                }

                if (radius <= 0.00001f)
                {
                    radius = Mathf.Max(
                        Mathf.Abs(segMax.x - segMin.x) * 0.5f,
                        Mathf.Abs(segMax.y - segMin.y) * 0.5f);
                }

                result.Add(new LocalCapsule
                {
                    a = new Vector3(centerX, centerY, z0),
                    b = new Vector3(centerX, centerY, z1),
                    radius = Mathf.Max(minRadius, radius * radiusScale)
                });
            }

            Debug.Log($"[GripperTool] {label} fitted with {result.Count} capsules " +
                      $"from collision mesh z=[{zMin:F3}, {zMax:F3}]");
        }

        void FitFallbackJawCapsules(List<LocalCapsule> result, int count)
        {
            float zMin = Mathf.Min(_pivotLocal.z, _tipLocal.z);
            float zMax = Mathf.Max(_pivotLocal.z, _tipLocal.z);
            float centerX = (_pivotLocal.x + _tipLocal.x) * 0.5f;
            float centerY = (_pivotLocal.y + _tipLocal.y) * 0.5f;

            for (int c = 0; c < count; c++)
            {
                result.Add(new LocalCapsule
                {
                    a = new Vector3(centerX, centerY, Mathf.Lerp(zMin, zMax, c / (float)count)),
                    b = new Vector3(centerX, centerY, Mathf.Lerp(zMin, zMax, (c + 1) / (float)count)),
                    radius = capsuleRadius
                });
            }
        }

        void FitShaftCapsuleFromCollisionMesh()
        {
            if (_colVertsShaft == null || _colVertsShaft.Length == 0)
                return;

            Vector3 min = _colVertsShaft[0];
            Vector3 max = _colVertsShaft[0];
            for (int i = 1; i < _colVertsShaft.Length; i++)
            {
                min = Vector3.Min(min, _colVertsShaft[i]);
                max = Vector3.Max(max, _colVertsShaft[i]);
            }

            float centerX = (min.x + max.x) * 0.5f;
            float centerY = (min.y + max.y) * 0.5f;
            _shaftEndLocal = new Vector3(centerX, centerY, max.z);
            _shaftTipLocal = new Vector3(centerX, centerY, min.z);

            Debug.Log($"[GripperTool] Shaft capsule fitted from collision mesh: " +
                      $"localZ=[{min.z:F3}, {max.z:F3}] center=({centerX:F3}, {centerY:F3}) radius={shaftRadius:F3}");
        }

        GameObject CreateOrReuseMeshObject(
            GameObject sceneObject,
            string name,
            string objFile,
            Material mat,
            out bool ownsObject)
        {
            var go = sceneObject;
            ownsObject = false;
            if (go == null)
            {
                /*
                Debug.LogError($"[GripperTool] 闁哄牜浜濈€垫氨鈧鑹惧┃鈧柡鍜佸灟閼垫垿鎯?{name}闁?, this);
                */
                Debug.LogError($"[GripperTool] Scene object is not assigned: {name}", this);
                return null;
            }

            var mf = go.GetComponent<MeshFilter>();
            var mr = go.GetComponent<MeshRenderer>();
            if (mf == null || mr == null)
            {
                /*
                Debug.LogError($"[GripperTool] {name} 闊洤鎳橀妴蹇旓紣閸曨偄甯ラ柟绋垮€藉ù?MeshFilter 闁?MeshRenderer闁?, go);
                */
                Debug.LogError($"[GripperTool] {name} requires MeshFilter and MeshRenderer components.", go);
                return go;
            }
            mr.material = mat;

            string path = Path.Combine(Application.streamingAssetsPath, objFile);
            if (File.Exists(path))
            {
                Mesh mesh = LoadOBJMesh(path);
                if (mesh != null)
                {
                    mf.mesh = mesh;
                    Debug.Log($"[GripperTool] Loaded {name}: {mesh.vertexCount}V");
                }
            }
            else
            {
                Debug.LogWarning($"[GripperTool] OBJ not found: {path}");
            }
            return go;
        }

        // 闁冲厜鍋撻柍鍏夊亾 闁告瑯鍨甸～瀣礌閺嶃劍绾柡?闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?
        void UpdateVisual()
        {
            Quaternion toolRot = Quaternion.Euler(0, _toolRotY, 0);
            transform.position = _toolPos;
            transform.rotation = toolRot;

            if (_shaftObj != null)
            {
                _shaftObj.transform.position = _toolPos;
                _shaftObj.transform.rotation = toolRot;
                _shaftObj.transform.localScale = Vector3.one * modelScale;
            }

            if (_jawUpObj != null)
            {
                // 濞戞挸锕。鍧楁晬濮橆剙甯ラ梺鍓у鐢挳寮€ｎ厽绁柨娑樼墕缁辨垿宕ラ崼顒傜闁挎稑鑻崯鈧柡浣虹節缂嶅寮€ｎ厽绁?妤犵偛纾簺
                // 闁?localScale=modelScale 濞戞挸顑戠槐婵嬫煣閻楀牆澶嶉柣鎰潐濡叉悂宕?OBJ 闁告鍠庨～鎰板锤閹邦厾鍨?
                _jawUpObj.transform.localScale = Vector3.one * modelScale;
                _jawUpObj.transform.position = _toolPos;
                _jawUpObj.transform.rotation = toolRot;
                ApplyJawLocalTransform(_jawUpObj, true, _currentAngle);

                // 闁?pivot 闁稿鑹剧槐鎴﹀触閸剛绐楅柛蹇撶墢浜涢柛?pivot闁挎稑鏈Λ鍡樻姜椤掑﹦绀夐柛鎰Ф浜涢柛?                // 缂佺姭鍋撻柛鏍ㄧ壄缁辨壆鈧絻顫夐弳锝嗙▔?jaw mesh 闁稿鑹捐ぐ澶愬箲?                ApplyJawRotation(_jawUpObj, true, _currentAngle, toolRot);
            }

            if (_jawDownObj != null)
            {
                _jawDownObj.transform.localScale = Vector3.one * modelScale;
                _jawDownObj.transform.position = _toolPos;
                _jawDownObj.transform.rotation = toolRot;

                ApplyJawLocalTransform(_jawDownObj, false, -_currentAngle);
            }
        }

        void ApplyJawLocalTransform(GameObject jawObj, bool isUpperJaw, float angle)
        {
            Vector3 selfAxisPointLocal = isUpperJaw
                ? _upperJawSelfAxisPointLocal
                : _lowerJawSelfAxisPointLocal;
            Quaternion jawMountRot = Quaternion.AngleAxis(jawRollAngle, Vector3.forward);
            Quaternion jawHingeRot = Quaternion.AngleAxis(angle, Vector3.right);
            Quaternion selfRollRot = Quaternion.AngleAxis(jawSelfRollAngle, Vector3.forward);

            // The jaw meshes are children of the tool root. This matches the local
            // transform used by collision and grasping geometry for both jaw sides.
            jawObj.transform.localRotation = jawMountRot * jawHingeRot * selfRollRot;
            jawObj.transform.localPosition = GetJawCenteringOffsetLocal(isUpperJaw) + _pivotLocal
                + jawMountRot * jawHingeRot
                * (selfAxisPointLocal - _pivotLocal - selfRollRot * selfAxisPointLocal);
            jawObj.transform.localScale = Vector3.one * modelScale;
        }

        void ApplyJawRotation(
            GameObject jawObj,
            bool isUpperJaw,
            float angle,
            Quaternion toolRot)
        {
            // 闂佸墽澧楃敮鎾倷閸︻厽鐣卞☉鎾寸墱閺咁偅鎷呭鍥╂瀭
            Vector3 pivotWorld = toolRot * _pivotLocal + _toolPos;
            Vector3 selfAxisPointLocal = isUpperJaw
                ? _upperJawSelfAxisPointLocal
                : _lowerJawSelfAxisPointLocal;
            Vector3 selfAxisPointWorld = toolRot * selfAxisPointLocal + _toolPos;
            Quaternion jawMountRot = Quaternion.AngleAxis(jawRollAngle, Vector3.forward);

            // 闁稿繐鐗愰鏇犵磾椤旂厧鐓傜€规悶鍎遍崣鎸庢媴瀹ュ洨鏋?
            jawObj.transform.position = _toolPos;
            jawObj.transform.rotation = toolRot;
            jawObj.transform.localScale = Vector3.one * modelScale;

            // Apply the jaw self-roll before the hinge rotation.
            jawObj.transform.RotateAround(
            // 闁稿繐鐗忕划顐﹀触閸曨喖娈板☉鎿冨幖缁虹偓娼壕瀣閺夌儐鍓ㄧ槐婵嬪礃瀹ュ洨鎼忛柡澶婃闁拌京鈧懓顦抽ˉ濠囧籍鐎ｎ厽绁柨娑樻湰濞撳爼宕ユ惔锝囨悘閻庣懓顦抽ˉ濠囧触鎼达絾鐣遍梺鍓у鐢瓨娼弶鎴犵；闁告艾鐗勯埀?            jawObj.transform.RotateAround(
                selfAxisPointWorld, toolRot * Vector3.forward, jawSelfRollAngle);
            jawObj.transform.RotateAround(pivotWorld, toolRot * Vector3.forward, jawRollAngle);
            Vector3 hingeAxisWorld = toolRot * (jawMountRot * Vector3.right);
            jawObj.transform.RotateAround(pivotWorld, hingeAxisWorld, angle);
            jawObj.transform.position += toolRot * GetJawCenteringOffsetLocal(isUpperJaw);
        }

        // 闁冲厜鍋撻柍鍏夊亾 OBJ 閻熸瑱绲鹃悗?闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?
        static Mesh LoadOBJMesh(string path)
        {
            LoadOBJData(path, out Vector3[] verts, out int[] tris);
            if (verts == null || verts.Length == 0) return null;
            var mesh = new Mesh { name = Path.GetFileNameWithoutExtension(path) };
            mesh.SetVertices(new List<Vector3>(verts));
            mesh.SetTriangles(tris, 0);
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }

        static void LoadOBJData(string path, out Vector3[] verts, out int[] tris)
        {
            verts = null; tris = null;
            if (!File.Exists(path)) return;

            var vertList = new List<Vector3>();
            var triList  = new List<int>();
            foreach (string line in File.ReadAllLines(path))
            {
                string s = line.Trim();
                if (s.StartsWith("v "))
                {
                    var p = s.Split(new[]{' '}, System.StringSplitOptions.RemoveEmptyEntries);
                    if (p.Length >= 4)
                        vertList.Add(new Vector3(
                            float.Parse(p[1], CultureInfo.InvariantCulture),
                            float.Parse(p[2], CultureInfo.InvariantCulture),
                            float.Parse(p[3], CultureInfo.InvariantCulture)));
                }
                else if (s.StartsWith("f "))
                {
                    var p  = s.Split(new[]{' '}, System.StringSplitOptions.RemoveEmptyEntries);
                    var fv = new List<int>();
                    for (int i = 1; i < p.Length; i++)
                        fv.Add(int.Parse(p[i].Split('/')[0], CultureInfo.InvariantCulture) - 1);
                    for (int i = 1; i < fv.Count - 1; i++)
                    { triList.Add(fv[0]); triList.Add(fv[i]); triList.Add(fv[i+1]); }
                }
            }
            verts = vertList.ToArray();
            tris = triList.ToArray();
        }

        void OnDestroy()
        {
            if (_ownsJawUpObject && _jawUpObj != null) Destroy(_jawUpObj);
            if (_ownsJawDownObject && _jawDownObj != null) Destroy(_jawDownObj);
            if (_ownsShaftObject && _shaftObj != null) Destroy(_shaftObj);
        }

        // 闁冲厜鍋撻柍鍏夊亾 Gizmo: 缁炬壆澧楅幐鎺楁嚄鐠虹儤鎶勫ù锝嗘尭瑜拌尙鎲撮崱妤€顕?闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?
        void OnDrawGizmos()
        {
            if (!_initialized || !showCapsuleGizmo) return;

            for (int i = 0; i < _dbgCapsuleCount; i++)
            {
                if (i < _dbgUpperCapsuleCount)
                    Gizmos.color = new Color(0f, 1f, 1f, 0.5f);
                else if (i < _dbgUpperCapsuleCount + _dbgLowerCapsuleCount)
                    Gizmos.color = new Color(1f, 0f, 1f, 0.5f);
                else
                    Gizmos.color = new Color(0f, 1f, 0f, 0.3f);

                DrawCapsuleGizmo(_dbgCapsuleA[i], _dbgCapsuleB[i], _dbgCapsuleR[i]);
            }

            // AABB 闁?濮掓稑瀚竟濠勭棯閹稿寒鏀?
            if (_dbgCapsuleCount > 0)
            {
                Gizmos.color = Color.yellow;
                Vector3 center = (_dbgBBoxMin + _dbgBBoxMax) * 0.5f;
                Vector3 size   = _dbgBBoxMax - _dbgBBoxMin;
                Gizmos.DrawWireCube(center, size);
            }
        }

        static void DrawCapsuleGizmo(Vector3 a, Vector3 b, float r)
        {
            Gizmos.DrawWireSphere(a, r);
            Gizmos.DrawWireSphere(b, r);
            Gizmos.DrawLine(a, b);
            Vector3 dir = (b - a).normalized;
            Vector3 up = Vector3.up;
            if (Mathf.Abs(Vector3.Dot(dir, up)) > 0.9f) up = Vector3.right;
            Vector3 side = Vector3.Cross(dir, up).normalized * r;
            Vector3 topDir = Vector3.Cross(dir, side).normalized * r;
            Gizmos.DrawLine(a + side, b + side);
            Gizmos.DrawLine(a - side, b - side);
            Gizmos.DrawLine(a + topDir, b + topDir);
            Gizmos.DrawLine(a - topDir, b - topDir);
        }

        // 闁冲厜鍋撻柍鍏夊亾 GUI 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾
        void OnGUI()
        {
            if (!_initialized) return;
            var style = new GUIStyle(GUI.skin.box) { fontSize = 12 };
            style.normal.textColor = Color.white;
            style.alignment = TextAnchor.UpperLeft;
            style.richText = true;

            string state = IsGrasping ? "<color=#FF4444>Grasping</color>" :
                          (_wantClose ? "<color=#FFFF00>Closing</color>" : "Open");

            string info =
                $"Gripper: {state}\n" +
                $"Angle: {_currentAngle:F1} deg\n" +
                $"Particles: core={_graspedParticles.Count} transition={_transitionParticleCount}\n" +
                $"Capsules: jaw={_dbgUpperCapsuleCount + _dbgLowerCapsuleCount} total={_dbgCapsuleCount}\n" +
                $"Contact candidates={_solver?.ActiveToolContactCandidateCount ?? 0} active={_solver?.ToolContactEnabled ?? false}\n" +
                "Move F/H G/B V/N | Rotate Numpad 1/3 | Close/Open Z/X";
            GUI.Box(new Rect(10, 360, 300, 128), info, style);
        }
    }
}

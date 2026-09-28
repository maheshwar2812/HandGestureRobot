import streamlit as st
import cv2
import mediapipe as mp
import math
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Vision Robot Control",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# AUTHENTICATION
# ============================================================

from auth import login_page, logout

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_page()
    st.stop()


# ============================================================
# SESSION VARIABLES
# ============================================================

if "robot_x" not in st.session_state:
    st.session_state.robot_x = 300

if "robot_y" not in st.session_state:
    st.session_state.robot_y = 250

if "current_command" not in st.session_state:
    st.session_state.current_command = "STOP"

if "emergency_stop" not in st.session_state:
    st.session_state.emergency_stop = False

if "command_history" not in st.session_state:
    st.session_state.command_history = []


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #0e1117;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #263244;
    }

    /* Sidebar title */
    .sidebar-title {
        font-size: 25px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .sidebar-subtitle {
        color: #9ca3af;
        font-size: 13px;
        margin-bottom: 25px;
    }

    /* Main title */
    .main-title {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 3px;
    }

    .main-subtitle {
        color: #9ca3af;
        font-size: 15px;
        margin-bottom: 25px;
    }

    /* Cards */
    .card {
        background: #151b26;
        border: 1px solid #263244;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 15px;
    }

    .card-title {
        font-size: 14px;
        color: #9ca3af;
        margin-bottom: 8px;
    }

    .card-value {
        font-size: 25px;
        font-weight: 700;
    }

    /* Command */
    .command-card {
        background: #151b26;
        border: 1px solid #263244;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        min-height: 230px;
    }

    .command-label {
        color: #9ca3af;
        font-size: 14px;
        margin-bottom: 15px;
    }

    .command-value {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .command-description {
        color: #9ca3af;
        font-size: 14px;
    }

    /* Status */
    .status-online {
        color: #22c55e;
        font-weight: 700;
    }

    .status-offline {
        color: #ef4444;
        font-weight: 700;
    }

    /* Section title */
    .section-title {
        font-size: 20px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    /* User card */
    .user-card {
        background: #151b26;
        border: 1px solid #263244;
        border-radius: 12px;
        padding: 15px;
        margin-top: 20px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 12px;
        padding: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-title">🤖 Vision Robot</div>
        <div class="sidebar-subtitle">
        Hand Gesture Control System
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🏠 Navigation")

    page = st.radio(
        "",
        [
            "Dashboard",
            "Live Control",
            "Robot",
            "Analytics",
            "History",
            "Settings"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("### 👤 User")

    st.write(f"**{st.session_state.get('user_name', 'User')}**")
    st.caption(st.session_state.get("user_email", ""))

    if st.button("🚪 Logout", use_container_width=True):
        logout()


# ============================================================
# MEDIAPIPE TASKS API
# ============================================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = mp.tasks.vision.HandLandmarker

HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

VisionRunningMode = mp.tasks.vision.RunningMode


# ============================================================
# HELPER FUNCTION
# ============================================================

def distance(a, b):

    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2
    )


# ============================================================
# GESTURE RECOGNITION
# ============================================================

def recognize_gesture(hand):

    wrist = hand[0]

    index_mcp = hand[5]
    index_pip = hand[6]
    index_tip = hand[8]

    middle_mcp = hand[9]
    middle_pip = hand[10]
    middle_tip = hand[12]

    ring_mcp = hand[13]
    ring_pip = hand[14]
    ring_tip = hand[16]

    little_mcp = hand[17]
    little_pip = hand[18]
    little_tip = hand[20]

    # --------------------------------------------------------
    # Finger extension detection
    # --------------------------------------------------------

    index_extended = (
        distance(index_tip, wrist)
        >
        distance(index_pip, wrist) * 1.12
    )

    middle_extended = (
        distance(middle_tip, wrist)
        >
        distance(middle_pip, wrist) * 1.12
    )

    ring_extended = (
        distance(ring_tip, wrist)
        >
        distance(ring_pip, wrist) * 1.12
    )

    little_extended = (
        distance(little_tip, wrist)
        >
        distance(little_pip, wrist) * 1.12
    )

    # --------------------------------------------------------
    # OPEN PALM
    # --------------------------------------------------------

    if (
        index_extended
        and middle_extended
        and ring_extended
        and little_extended
    ):

        return "STOP"

    # --------------------------------------------------------
    # INDEX FINGER ONLY
    # --------------------------------------------------------

    if (
        index_extended
        and not middle_extended
        and not ring_extended
        and not little_extended
    ):

        dx = index_tip.x - index_mcp.x
        dy = index_tip.y - index_mcp.y

        # Vertical movement
        if abs(dy) > abs(dx):

            if dy < -0.08:
                return "FORWARD"

            elif dy > 0.08:
                return "BACKWARD"

        # Horizontal movement
        else:

            if dx < -0.08:
                return "LEFT"

            elif dx > 0.08:
                return "RIGHT"

    return "UNKNOWN"


# ============================================================
# COMMAND INFORMATION
# ============================================================

COMMAND_INFO = {

    "FORWARD": {
        "icon": "⬆️",
        "description": "Robot moving forward"
    },

    "BACKWARD": {
        "icon": "⬇️",
        "description": "Robot moving backward"
    },

    "LEFT": {
        "icon": "⬅️",
        "description": "Robot turning left"
    },

    "RIGHT": {
        "icon": "➡️",
        "description": "Robot turning right"
    },

    "STOP": {
        "icon": "🛑",
        "description": "Robot stopped"
    },

    "UNKNOWN": {
        "icon": "❔",
        "description": "Gesture not recognized"
    }
}


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class VideoProcessor(VideoProcessorBase):

    def __init__(self):

        self.model_path = "models/hand_landmarker.task"

        options = HandLandmarkerOptions(

            base_options=BaseOptions(
                model_asset_path=self.model_path
            ),

            running_mode=VisionRunningMode.IMAGE,

            num_hands=1
        )

        self.landmarker = HandLandmarker.create_from_options(
            options
        )

        self.gesture = "STOP"

        self.last_time = time.time()

        self.fps = 0

    # --------------------------------------------------------
    # PROCESS FRAME
    # --------------------------------------------------------

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        # Mirror camera
        img = cv2.flip(img, 1)

        rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = self.landmarker.detect(mp_image)

        detected_gesture = "UNKNOWN"

        # ----------------------------------------------------
        # HAND DETECTION
        # ----------------------------------------------------

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            height, width, _ = img.shape

            # Draw landmarks
            for landmark in hand:

                x = int(landmark.x * width)

                y = int(landmark.y * height)

                cv2.circle(
                    img,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            # Draw connections manually
            connections = [

                (0, 1), (1, 2), (2, 3), (3, 4),

                (0, 5), (5, 6), (6, 7), (7, 8),

                (0, 9), (9, 10), (10, 11), (11, 12),

                (0, 13), (13, 14), (14, 15), (15, 16),

                (0, 17), (17, 18), (18, 19), (19, 20),

                (5, 9), (9, 13), (13, 17)

            ]

            for start, end in connections:

                x1 = int(hand[start].x * width)
                y1 = int(hand[start].y * height)

                x2 = int(hand[end].x * width)
                y2 = int(hand[end].y * height)

                cv2.line(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

            detected_gesture = recognize_gesture(hand)

        else:

            detected_gesture = "STOP"

        # ----------------------------------------------------
        # EMERGENCY STOP
        # ----------------------------------------------------

        if st.session_state.get("emergency_stop", False):

            detected_gesture = "STOP"

        self.gesture = detected_gesture

        # ----------------------------------------------------
        # ROBOT MOVEMENT
        # ----------------------------------------------------

        if detected_gesture == "FORWARD":

            st.session_state.robot_y -= 3

        elif detected_gesture == "BACKWARD":

            st.session_state.robot_y += 3

        elif detected_gesture == "LEFT":

            st.session_state.robot_x -= 3

        elif detected_gesture == "RIGHT":

            st.session_state.robot_x += 3

        # ----------------------------------------------------
        # ROBOT BOUNDARIES
        # ----------------------------------------------------

        st.session_state.robot_x = max(
            80,
            min(
                st.session_state.robot_x,
                width - 80
            )
        )

        st.session_state.robot_y = max(
            80,
            min(
                st.session_state.robot_y,
                height - 80
            )
        )

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        current_time = time.time()

        elapsed = current_time - self.last_time

        if elapsed > 0:

            self.fps = int(1 / elapsed)

        self.last_time = current_time

        # ----------------------------------------------------
        # CAMERA COMMAND DISPLAY
        # ----------------------------------------------------

        command_info = COMMAND_INFO.get(
            detected_gesture,
            COMMAND_INFO["UNKNOWN"]
        )

        text = (
            f"{command_info['icon']} "
            f"{detected_gesture}"
        )

        cv2.putText(
            img,
            text,
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 0),
            3
        )

        # FPS
        cv2.putText(
            img,
            f"FPS: {self.fps}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        return frame.from_ndarray(
            img,
            format="bgr24"
        )


# ============================================================
# ROBOT DRAWING
# ============================================================

def draw_robot():

    width = 700
    height = 450

    robot_x = int(st.session_state.robot_x)
    robot_y = int(st.session_state.robot_y)

    canvas = (
        15 *
        __import__("numpy").ones(
            (height, width, 3),
            dtype="uint8"
        )
    )

    canvas = canvas.astype("uint8")

    # Background
    canvas[:] = (20, 25, 35)

    # Grid
    for x in range(0, width, 50):

        cv2.line(
            canvas,
            (x, 0),
            (x, height),
            (45, 50, 60),
            1
        )

    for y in range(0, height, 50):

        cv2.line(
            canvas,
            (0, y),
            (width, y),
            (45, 50, 60),
            1
        )

    # Robot body
    cv2.rectangle(
        canvas,
        (robot_x - 45, robot_y - 35),
        (robot_x + 45, robot_y + 35),
        (60, 130, 220),
        -1
    )

    # Robot head
    cv2.rectangle(
        canvas,
        (robot_x - 35, robot_y - 75),
        (robot_x + 35, robot_y - 35),
        (80, 150, 230),
        -1
    )

    # Eyes
    cv2.circle(
        canvas,
        (robot_x - 15, robot_y - 55),
        6,
        (255, 255, 255),
        -1
    )

    cv2.circle(
        canvas,
        (robot_x + 15, robot_y - 55),
        6,
        (255, 255, 255),
        -1
    )

    # Wheels
    cv2.circle(
        canvas,
        (robot_x - 45, robot_y + 45),
        14,
        (100, 100, 100),
        -1
    )

    cv2.circle(
        canvas,
        (robot_x + 45, robot_y + 45),
        14,
        (100, 100, 100),
        -1
    )

    # Antenna
    cv2.line(
        canvas,
        (robot_x, robot_y - 75),
        (robot_x, robot_y - 100),
        (200, 200, 200),
        3
    )

    cv2.circle(
        canvas,
        (robot_x, robot_y - 105),
        6,
        (0, 255, 0),
        -1
    )

    # Current command
    command = st.session_state.current_command

    cv2.putText(
        canvas,
        f"COMMAND: {command}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    return canvas


# ============================================================
# DASHBOARD PAGE
# ============================================================

if page == "Dashboard":

    st.markdown(
        '<div class="main-title">🤖 Vision Robot Control</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="main-subtitle">'
        'Real-time hand gesture controlled virtual robot'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # TOP STATUS CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            """
            <div class="card">
                <div class="card-title">SYSTEM STATUS</div>
                <div class="card-value">
                    🟢 ACTIVE
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="card">
                <div class="card-title">CAMERA</div>
                <div class="card-value">
                    📷 READY
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="card">
                <div class="card-title">HAND TRACKING</div>
                <div class="card-value">
                    🖐️ READY
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            """
            <div class="card">
                <div class="card-title">CONTROL MODE</div>
                <div class="card-value">
                    🎮 GESTURE
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-title">🎮 Live Robot Control</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # CAMERA + COMMAND
    # --------------------------------------------------------

    camera_col, command_col = st.columns([2, 1])

    with camera_col:

        st.markdown("### 📷 Camera")

        webrtc_ctx = webrtc_streamer(

            key="robot-camera",

            video_processor_factory=VideoProcessor,

            media_stream_constraints={
                "video": True,
                "audio": False
            },

            async_processing=True
        )

    with command_col:

        command = st.session_state.current_command

        info = COMMAND_INFO.get(
            command,
            COMMAND_INFO["UNKNOWN"]
        )

        st.markdown(
            f"""
            <div class="command-card">

                <div class="command-label">
                    CURRENT ROBOT COMMAND
                </div>

                <div class="command-value">
                    {info["icon"]} {command}
                </div>

                <div class="command-description">
                    {info["description"]}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        if st.button(
            "🛑 EMERGENCY STOP",
            use_container_width=True,
            type="primary"
        ):

            st.session_state.emergency_stop = True

            st.session_state.current_command = "STOP"

            st.rerun()

        if st.button(
            "▶️ Resume Control",
            use_container_width=True
        ):

            st.session_state.emergency_stop = False

            st.rerun()

    # --------------------------------------------------------
    # VIRTUAL ROBOT
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🤖 Virtual Robot</div>',
        unsafe_allow_html=True
    )

    robot_col, control_col = st.columns([2, 1])

    with robot_col:

        robot_image = draw_robot()

        st.image(
            robot_image,
            use_container_width=True
        )

    with control_col:

        st.markdown("### Robot Status")

        if st.session_state.emergency_stop:

            st.error("🛑 EMERGENCY STOP ACTIVE")

        else:

            st.success("🟢 ROBOT ACTIVE")

        st.metric(
            "X Position",
            int(st.session_state.robot_x)
        )

        st.metric(
            "Y Position",
            int(st.session_state.robot_y)
        )

        if st.button(
            "🔄 Reset Robot Position",
            use_container_width=True
        ):

            st.session_state.robot_x = 300
            st.session_state.robot_y = 250

            st.rerun()


# ============================================================
# LIVE CONTROL PAGE
# ============================================================

elif page == "Live Control":

    st.markdown(
        '<div class="main-title">📷 Live Control</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="main-subtitle">'
        'Control the virtual robot using hand gestures'
        '</div>',
        unsafe_allow_html=True
    )

    webrtc_streamer(

        key="live-control-camera",

        video_processor_factory=VideoProcessor,

        media_stream_constraints={
            "video": True,
            "audio": False
        },

        async_processing=True
    )

    st.info(
        "☝️ Forward   |   👇 Backward   |   👈 Left   |   👉 Right   |   ✋ Stop"
    )


# ============================================================
# ROBOT PAGE
# ============================================================

elif page == "Robot":

    st.markdown(
        '<div class="main-title">🤖 Robot</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="main-subtitle">'
        'Virtual robot monitoring and control'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([2, 1])

    with col1:

        st.image(
            draw_robot(),
            use_container_width=True
        )

    with col2:

        st.markdown("### Robot Information")

        st.write(
            f"**X Position:** {int(st.session_state.robot_x)}"
        )

        st.write(
            f"**Y Position:** {int(st.session_state.robot_y)}"
        )

        st.write(
            f"**Command:** {st.session_state.current_command}"
        )

        if st.session_state.emergency_stop:

            st.error("Emergency Stop Active")

        else:

            st.success("Robot Operational")

        if st.button(
            "🔄 Reset Robot",
            use_container_width=True
        ):

            st.session_state.robot_x = 300
            st.session_state.robot_y = 250

            st.rerun()


# ============================================================
# ANALYTICS PAGE
# ============================================================

elif page == "Analytics":

    st.markdown(
        '<div class="main-title">📊 Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="main-subtitle">'
        'Robot control system statistics'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Commands",
            len(st.session_state.command_history)
        )

    with col2:

        st.metric(
            "Current Command",
            st.session_state.current_command
        )

    with col3:

        st.metric(
            "Robot Status",
            "STOPPED"
            if st.session_state.emergency_stop
            else "ACTIVE"
        )

    st.markdown("### Gesture Classes")

    st.write(
        """
        The system currently recognizes:

        - ☝️ FORWARD
        - 👇 BACKWARD
        - 👈 LEFT
        - 👉 RIGHT
        - ✋ STOP
        """
    )


# ============================================================
# HISTORY PAGE
# ============================================================

elif page == "History":

    st.markdown(
        '<div class="main-title">📜 Command History</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="main-subtitle">'
        'Recently detected robot commands'
        '</div>',
        unsafe_allow_html=True
    )

    if len(st.session_state.command_history) == 0:

        st.info(
            "No command history available yet."
        )

    else:

        for command in reversed(
            st.session_state.command_history
        ):

            info = COMMAND_INFO.get(
                command,
                COMMAND_INFO["UNKNOWN"]
            )

            st.write(
                f"{info['icon']} **{command}**"
            )


# ============================================================
# SETTINGS PAGE
# ============================================================

elif page == "Settings":

    st.markdown(
        '<div class="main-title">⚙️ Settings</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="main-subtitle">'
        'Application configuration'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("### 🎮 Robot Settings")

    speed = st.slider(
        "Robot Speed",
        min_value=1,
        max_value=10,
        value=3
    )

    st.write(
        f"Current robot speed: **{speed} px/frame**"
    )

    st.markdown("### 📷 Camera Settings")

    mirror_camera = st.checkbox(
        "Mirror Camera",
        value=True
    )

    show_landmarks = st.checkbox(
        "Show Hand Landmarks",
        value=True
    )

    st.markdown("### 🛡️ Safety")

    safety_mode = st.toggle(
        "Enable Safety Mode",
        value=True
    )

    if safety_mode:

        st.success(
            "Safety mode enabled"
        )

    else:

        st.warning(
            "Safety mode disabled"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Vision-Based Hand Gesture Recognition for Real-Time Robot Control
        <br>
        OpenCV • MediaPipe • Streamlit • Python
    </div>
    """,
    unsafe_allow_html=True
)
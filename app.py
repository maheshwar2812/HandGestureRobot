import streamlit as st
import streamlit.components.v1 as components

import cv2
import av
import mediapipe as mp

from streamlit_webrtc import webrtc_streamer

from threading import Lock
from datetime import datetime


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="HandGestureRobot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main application */

    .stApp {
        background: #0b1120;
    }


    /* Sidebar */

    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #1f2937;
    }


    /* Headings */

    h1 {
        color: #f8fafc !important;
        font-weight: 700 !important;
    }

    h2, h3 {
        color: #e5e7eb !important;
    }


    /* Text */

    p {
        color: #cbd5e1;
    }


    /* Metric cards */

    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #263244;
        border-radius: 12px;
        padding: 14px;
    }


    /* Buttons */

    .stButton > button {
        border-radius: 8px;
        min-height: 42px;
        font-weight: 600;
    }


    /* Status card */

    .status-card {
        background: #111827;
        border: 1px solid #263244;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        margin-bottom: 10px;
    }


    /* Command card */

    .command-card {
        background: #111827;
        border: 1px solid #263244;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        margin-top: 10px;
    }


    /* Section card */

    .section-card {
        background: #111827;
        border: 1px solid #263244;
        border-radius: 14px;
        padding: 18px;
    }


    /* Small text */

    .small-text {
        color: #94a3b8;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
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
# THREAD-SAFE COMMAND STATE
# ============================================================

class CommandState:

    def __init__(self):

        self.command = "STOP"

        self.lock = Lock()

    def set_command(self, command):

        with self.lock:

            self.command = command

    def get_command(self):

        with self.lock:

            return self.command


if "command_state" not in st.session_state:

    st.session_state.command_state = CommandState()


command_state = st.session_state.command_state


# ============================================================
# SESSION STATE
# ============================================================

defaults = {

    "robot_x": 50.0,

    "robot_y": 50.0,

    "robot_speed": 1.5,

    "emergency_stop": False,

    "history": [],

    "last_recorded_command": None,

    "total_commands": 0,

    "gesture_counts": {

        "FORWARD": 0,

        "BACKWARD": 0,

        "LEFT": 0,

        "RIGHT": 0,

        "STOP": 0
    },

    "robot_path": [],

    "show_landmarks": True,

    "mirror_camera": True
}


for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# GESTURE RECOGNITION
# ============================================================

def recognize_gesture(hand):

    fingers_up = 0


    # Index

    if hand[8].y < hand[6].y:
        fingers_up += 1


    # Middle

    if hand[12].y < hand[10].y:
        fingers_up += 1


    # Ring

    if hand[16].y < hand[14].y:
        fingers_up += 1


    # Little

    if hand[20].y < hand[18].y:
        fingers_up += 1


    # Gesture mapping

    if fingers_up == 0:

        return "STOP"

    elif fingers_up == 1:

        return "FORWARD"

    elif fingers_up == 2:

        return "BACKWARD"

    elif fingers_up == 3:

        return "LEFT"

    elif fingers_up == 4:

        return "RIGHT"


    return "STOP"


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = mp.tasks.vision.HandLandmarker

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)


options = HandLandmarkerOptions(

    base_options=BaseOptions(

        model_asset_path="models/hand_landmarker.task"
    ),

    running_mode=VisionRunningMode.IMAGE,

    num_hands=1
)


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class VideoProcessor:

    def __init__(self):

        self.landmarker = (
            HandLandmarker.create_from_options(
                options
            )
        )


    def recv(self, frame):

        # ----------------------------------------------------
        # Convert frame
        # ----------------------------------------------------

        img = frame.to_ndarray(
            format="bgr24"
        )


        # ----------------------------------------------------
        # Mirror
        # ----------------------------------------------------

        if st.session_state.get(
            "mirror_camera",
            True
        ):

            img = cv2.flip(
                img,
                1
            )


        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        rgb_img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # MediaPipe image
        # ----------------------------------------------------

        mp_image = mp.Image(

            image_format=mp.ImageFormat.SRGB,

            data=rgb_img
        )


        # ----------------------------------------------------
        # Detect
        # ----------------------------------------------------

        result = self.landmarker.detect(
            mp_image
        )


        gesture = "NO HAND"


        # ----------------------------------------------------
        # Hand detected
        # ----------------------------------------------------

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]


            # Recognize gesture

            gesture = recognize_gesture(
                hand
            )


            # Send command

            command_state.set_command(
                gesture
            )


            # Draw landmarks

            if st.session_state.get(
                "show_landmarks",
                True
            ):

                height, width, _ = img.shape


                for landmark in hand:

                    x = int(
                        landmark.x * width
                    )

                    y = int(
                        landmark.y * height
                    )


                    cv2.circle(

                        img,

                        (x, y),

                        5,

                        (0, 255, 0),

                        -1
                    )


        else:

            # Safety

            command_state.set_command(
                "STOP"
            )


        # ----------------------------------------------------
        # Command color
        # ----------------------------------------------------

        if gesture == "FORWARD":

            command_color = (0, 255, 0)

        elif gesture == "BACKWARD":

            command_color = (0, 200, 255)

        elif gesture == "LEFT":

            command_color = (255, 200, 0)

        elif gesture == "RIGHT":

            command_color = (255, 0, 255)

        else:

            command_color = (0, 0, 255)


        # ----------------------------------------------------
        # Command box
        # ----------------------------------------------------

        cv2.rectangle(

            img,

            (15, 15),

            (440, 90),

            (10, 15, 25),

            -1
        )


        cv2.putText(

            img,

            "COMMAND",

            (30, 45),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (180, 190, 200),

            2
        )


        cv2.putText(

            img,

            gesture,

            (170, 67),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.85,

            command_color,

            3
        )


        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

        return av.VideoFrame.from_ndarray(

            img,

            format="bgr24"
        )


# ============================================================
# ROBOT MOVEMENT
# ============================================================

def move_robot(command):

    if st.session_state.emergency_stop:

        return


    speed = st.session_state.robot_speed


    old_x = st.session_state.robot_x

    old_y = st.session_state.robot_y


    if command == "FORWARD":

        st.session_state.robot_y -= speed


    elif command == "BACKWARD":

        st.session_state.robot_y += speed


    elif command == "LEFT":

        st.session_state.robot_x -= speed


    elif command == "RIGHT":

        st.session_state.robot_x += speed


    # --------------------------------------------------------
    # Boundary
    # --------------------------------------------------------

    st.session_state.robot_x = max(

        8,

        min(
            92,
            st.session_state.robot_x
        )
    )


    st.session_state.robot_y = max(

        10,

        min(
            90,
            st.session_state.robot_y
        )
    )


    # --------------------------------------------------------
    # Add path point only when moving
    # --------------------------------------------------------

    if command in [

        "FORWARD",

        "BACKWARD",

        "LEFT",

        "RIGHT"

    ]:

        if (

            old_x != st.session_state.robot_x

            or

            old_y != st.session_state.robot_y

        ):

            st.session_state.robot_path.append(

                (
                    st.session_state.robot_x,

                    st.session_state.robot_y
                )
            )


    # Keep path small

    if len(
        st.session_state.robot_path
    ) > 100:

        st.session_state.robot_path = (
            st.session_state.robot_path[-100:]
        )


# ============================================================
# COMMAND HISTORY
# ============================================================

def record_command(command):

    if command == "NO HAND":

        return


    # Only record when command changes

    if (
        st.session_state.last_recorded_command
        == command
    ):

        return


    st.session_state.last_recorded_command = command


    st.session_state.total_commands += 1


    st.session_state.gesture_counts[
        command
    ] += 1


    now = datetime.now().strftime(
        "%H:%M:%S"
    )


    st.session_state.history.append({

        "Time": now,

        "Command": command,

        "X": round(
            st.session_state.robot_x,
            1
        ),

        "Y": round(
            st.session_state.robot_y,
            1
        )
    })


    # Keep latest 100

    if len(
        st.session_state.history
    ) > 100:

        st.session_state.history = (
            st.session_state.history[-100:]
        )


# ============================================================
# ROBOT DISPLAY
# ============================================================

def display_robot():

    x = st.session_state.robot_x

    y = st.session_state.robot_y


    # --------------------------------------------------------
    # Path SVG
    # --------------------------------------------------------

    path_svg = ""


    path = st.session_state.robot_path


    if len(path) >= 2:

        points = " ".join(

            f"{px},{py}"

            for px, py in path
        )


        path_svg = f"""

        <polyline

            points="{points}"

            fill="none"

            stroke="#38bdf8"

            stroke-width="0.7"

            stroke-dasharray="2,2"

            opacity="0.7"

        />

        """


    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    command = command_state.get_command()


    if command == "FORWARD":

        arrow = """
        <polygon
            points="50,8 46,15 54,15"
            fill="#22c55e"
        />
        """

    elif command == "BACKWARD":

        arrow = """
        <polygon
            points="50,92 46,85 54,85"
            fill="#f59e0b"
        />
        """

    elif command == "LEFT":

        arrow = """
        <polygon
            points="8,50 15,46 15,54"
            fill="#38bdf8"
        />
        """

    elif command == "RIGHT":

        arrow = """
        <polygon
            points="92,50 85,46 85,54"
            fill="#a78bfa"
        />
        """

    else:

        arrow = ""


    # --------------------------------------------------------
    # Emergency overlay
    # --------------------------------------------------------

    emergency = ""


    if st.session_state.emergency_stop:

        emergency = """

        <rect
            x="1"
            y="1"
            width="98"
            height="98"
            rx="3"
            fill="#7f1d1d"
            opacity="0.35"
        />

        <text
            x="50"
            y="48"
            text-anchor="middle"
            fill="#fecaca"
            font-size="6"
            font-weight="bold"
        >
            EMERGENCY STOP
        </text>

        """


    # --------------------------------------------------------
    # SVG
    # --------------------------------------------------------

    svg = f"""

    <svg

        viewBox="0 0 100 100"

        xmlns="http://www.w3.org/2000/svg"

        style="

            width:100%;

            height:430px;

            background:#0f172a;

            border:2px solid #334155;

            border-radius:16px;

        "
    >


        <!-- GRID -->

        <defs>

            <pattern

                id="grid"

                width="10"

                height="10"

                patternUnits="userSpaceOnUse"

            >

                <path

                    d="M 10 0 L 0 0 0 10"

                    fill="none"

                    stroke="#1e293b"

                    stroke-width="0.5"

                />

            </pattern>

        </defs>


        <rect

            width="100"

            height="100"

            fill="url(#grid)"

        />


        <!-- PATH -->

        {path_svg}


        <!-- START -->

        <circle
            cx="7"
            cy="92"
            r="2"
            fill="#22c55e"
        />


        <text
            x="11"
            y="94"
            fill="#94a3b8"
            font-size="3.5"
        >
            START
        </text>


        <!-- ROBOT -->

        <g transform="translate({x},{y})">


            <!-- Shadow -->

            <ellipse

                cx="0"

                cy="9"

                rx="10"

                ry="3"

                fill="#000000"

                opacity="0.35"

            />


            <!-- Wheels -->

            <rect

                x="-10"

                y="-1"

                width="4"

                height="10"

                rx="1"

                fill="#020617"

            />


            <rect

                x="6"

                y="-1"

                width="4"

                height="10"

                rx="1"

                fill="#020617"

            />


            <!-- Body -->

            <rect

                x="-7"

                y="-6"

                width="14"

                height="14"

                rx="3"

                fill="#2563eb"

                stroke="#60a5fa"

                stroke-width="1"

            />


            <!-- Head -->

            <rect

                x="-6"

                y="-13"

                width="12"

                height="7"

                rx="2"

                fill="#3b82f6"

                stroke="#93c5fd"

                stroke-width="0.8"

            />


            <!-- Eyes -->

            <circle

                cx="-2.5"

                cy="-9.5"

                r="1"

                fill="#ffffff"

            />


            <circle

                cx="2.5"

                cy="-9.5"

                r="1"

                fill="#ffffff"

            />


            <!-- Antenna -->

            <line

                x1="0"

                y1="-13"

                x2="0"

                y2="-17"

                stroke="#94a3b8"

                stroke-width="0.8"

            />


            <circle

                cx="0"

                cy="-18"

                r="1.3"

                fill="#22c55e"

            />


        </g>


        <!-- DIRECTION -->

        {arrow}


        <!-- EMERGENCY -->

        {emergency}


        <!-- BORDER -->

        <rect

            x="1"

            y="1"

            width="98"

            height="98"

            rx="3"

            fill="none"

            stroke="#475569"

            stroke-width="0.5"

        />


    </svg>

    """


    components.html(

        svg,

        height=450
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🤖 HandGestureRobot"
    )

    st.caption(
        "Vision-Based Real-Time Robot Control"
    )


    st.divider()


    # User

    st.markdown(
        "### 👤 Account"
    )


    st.write(
        st.session_state.get(
            "user_name",
            "User"
        )
    )


    st.caption(
        st.session_state.get(
            "user_email",
            ""
        )
    )


    st.divider()


    # Navigation

    page = st.radio(

        "Navigation",

        [

            "🏠 Dashboard",

            "📷 Live Control",

            "🤖 Robot",

            "📊 Analytics",

            "📜 History",

            "⚙️ Settings"

        ]
    )


    st.divider()


    # Emergency

    if st.button(

        "🚨 EMERGENCY STOP",

        use_container_width=True
    ):

        st.session_state.emergency_stop = True

        command_state.set_command(
            "STOP"
        )

        st.rerun()


    if st.session_state.emergency_stop:

        if st.button(

            "▶️ RESUME ROBOT",

            use_container_width=True
        ):

            st.session_state.emergency_stop = False

            command_state.set_command(
                "STOP"
            )

            st.rerun()


    st.divider()


    if st.button(

        "🚪 Logout",

        use_container_width=True
    ):

        logout()


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title(
        "🏠 Control Dashboard"
    )


    st.markdown(
        """
        ### Welcome to HandGestureRobot 👋

        A real-time computer vision system that uses hand gestures
        to control a virtual robot.
        """
    )


    st.divider()


    # Metrics

    command = command_state.get_command()


    m1, m2, m3, m4 = st.columns(4)


    with m1:

        st.metric(
            "Current Command",
            command
        )


    with m2:

        st.metric(
            "Commands",
            st.session_state.total_commands
        )


    with m3:

        st.metric(
            "Robot X",
            f"{st.session_state.robot_x:.0f}%"
        )


    with m4:

        st.metric(
            "Robot Y",
            f"{st.session_state.robot_y:.0f}%"
        )


    st.divider()


    # Status

    st.subheader(
        "📡 System Status"
    )


    a, b, c = st.columns(3)


    with a:

        st.success(
            "🟢 MediaPipe Ready"
        )


    with b:

        st.success(
            "🟢 Gesture Recognition Ready"
        )


    with c:

        if st.session_state.emergency_stop:

            st.error(
                "🔴 Emergency Stop"
            )

        else:

            st.success(
                "🟢 Robot Ready"
            )


    st.divider()


    # Gesture guide

    st.subheader(
        "🎮 Gesture Control"
    )


    g1, g2, g3, g4, g5 = st.columns(5)


    with g1:

        st.markdown(
            "### ☝️\n**FORWARD**\n\n1 Finger"
        )


    with g2:

        st.markdown(
            "### ✌️\n**BACKWARD**\n\n2 Fingers"
        )


    with g3:

        st.markdown(
            "### 🤟\n**LEFT**\n\n3 Fingers"
        )


    with g4:

        st.markdown(
            "### 🖐️\n**RIGHT**\n\n4 Fingers"
        )


    with g5:

        st.markdown(
            "### ✊\n**STOP**\n\n0 Fingers"
        )


# ============================================================
# LIVE CONTROL
# ============================================================

elif page == "📷 Live Control":

    st.title(
        "📷 Live Robot Control"
    )


    st.caption(
        "Show your hand to the camera and control the virtual robot."
    )


    # ========================================================
    # MAIN SPLIT SCREEN
    # ========================================================

    camera_col, robot_col = st.columns(
        [1, 1],
        gap="large"
    )


    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    with camera_col:

        st.subheader(
            "📷 Live Camera"
        )


        st.caption(
            "Hand detection + gesture recognition"
        )


        webrtc_streamer(

            key="main-live-camera",

            video_processor_factory=VideoProcessor,

            media_stream_constraints={

                "video": True,

                "audio": False
            },

            async_processing=True
        )


        # Current command

        current = command_state.get_command()


        st.markdown(
            f"""
            <div class="command-card">

            <div class="small-text">
            DETECTED COMMAND
            </div>

            <h2>{current}</h2>

            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # ROBOT
    # --------------------------------------------------------

    with robot_col:

        st.subheader(
            "🤖 Virtual Robot"
        )


        st.caption(
            "Real-time robot simulation"
        )


        robot_placeholder = st.empty()


        @st.fragment(run_every="200ms")
        def live_robot():

            current_command = (
                command_state.get_command()
            )


            # Move

            move_robot(
                current_command
            )


            # History

            record_command(
                current_command
            )


            # Robot

            with robot_placeholder:

                display_robot()


            # ------------------------------------------------
            # Status
            # ------------------------------------------------

            if st.session_state.emergency_stop:

                st.error(
                    "🚨 EMERGENCY STOP ACTIVE"
                )

            else:

                if current_command == "FORWARD":

                    st.success(
                        "⬆️ Moving Forward"
                    )

                elif current_command == "BACKWARD":

                    st.warning(
                        "⬇️ Moving Backward"
                    )

                elif current_command == "LEFT":

                    st.info(
                        "⬅️ Moving Left"
                    )

                elif current_command == "RIGHT":

                    st.info(
                        "➡️ Moving Right"
                    )

                else:

                    st.info(
                        "✋ Robot Stopped"
                    )


        live_robot()


    # ========================================================
    # CONTROL INFORMATION
    # ========================================================

    st.divider()


    p1, p2, p3, p4 = st.columns(4)


    with p1:

        st.metric(
            "X Position",
            f"{st.session_state.robot_x:.1f}%"
        )


    with p2:

        st.metric(
            "Y Position",
            f"{st.session_state.robot_y:.1f}%"
        )


    with p3:

        st.metric(
            "Speed",
            f"{st.session_state.robot_speed:.1f}"
        )


    with p4:

        st.metric(
            "Commands",
            st.session_state.total_commands
        )


    # Reset

    if st.button(
        "🔄 Reset Robot",
        use_container_width=True
    ):

        st.session_state.robot_x = 50

        st.session_state.robot_y = 50

        st.session_state.robot_path = []

        st.session_state.emergency_stop = False

        command_state.set_command(
            "STOP"
        )

        st.session_state.last_recorded_command = None

        st.rerun()


# ============================================================
# ROBOT PAGE
# ============================================================

elif page == "🤖 Robot":

    st.title(
        "🤖 Robot Simulator"
    )


    st.caption(
        "Detailed virtual robot simulation."
    )


    placeholder = st.empty()


    @st.fragment(run_every="200ms")
    def robot_page():

        command = command_state.get_command()


        move_robot(
            command
        )


        with placeholder:

            display_robot()


        st.markdown(
            f"### Current Command: `{command}`"
        )


    robot_page()


# ============================================================
# ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.title(
        "📊 Gesture Analytics"
    )


    counts = st.session_state.gesture_counts


    total = st.session_state.total_commands


    if total > 0:

        most_used = max(
            counts,
            key=counts.get
        )

    else:

        most_used = "NONE"


    a1, a2, a3 = st.columns(3)


    with a1:

        st.metric(
            "Total Commands",
            total
        )


    with a2:

        st.metric(
            "Most Used",
            most_used
        )


    with a3:

        st.metric(
            "Robot Position",
            f"{st.session_state.robot_x:.0f}% , "
            f"{st.session_state.robot_y:.0f}%"
        )


    st.divider()


    st.subheader(
        "📈 Gesture Frequency"
    )


    st.bar_chart(
        counts
    )


    st.divider()


    st.subheader(
        "Command Statistics"
    )


    for command, count in counts.items():

        st.write(
            f"**{command}** : {count}"
        )


# ============================================================
# HISTORY
# ============================================================

elif page == "📜 History":

    st.title(
        "📜 Command History"
    )


    st.caption(
        "Recent gesture commands detected by the system."
    )


    if not st.session_state.history:

        st.info(
            "No command history yet. Start the Live Control camera."
        )

    else:

        data = list(
            reversed(
                st.session_state.history
            )
        )


        st.dataframe(

            data,

            use_container_width=True,

            hide_index=True
        )


        st.divider()


        if st.button(
            "🗑️ Clear History"
        ):

            st.session_state.history = []

            st.session_state.gesture_counts = {

                "FORWARD": 0,

                "BACKWARD": 0,

                "LEFT": 0,

                "RIGHT": 0,

                "STOP": 0
            }

            st.session_state.total_commands = 0

            st.session_state.last_recorded_command = None

            st.rerun()


# ============================================================
# SETTINGS
# ============================================================

elif page == "⚙️ Settings":

    st.title(
        "⚙️ Settings"
    )


    # ========================================================
    # ROBOT
    # ========================================================

    st.subheader(
        "🤖 Robot Settings"
    )


    st.session_state.robot_speed = st.slider(

        "Robot Movement Speed",

        min_value=0.5,

        max_value=5.0,

        value=float(
            st.session_state.robot_speed
        ),

        step=0.5
    )


    st.caption(
        "Higher value = faster virtual robot movement."
    )


    st.divider()


    # ========================================================
    # CAMERA
    # ========================================================

    st.subheader(
        "📷 Camera Settings"
    )


    st.session_state.mirror_camera = st.checkbox(

        "Mirror Camera",

        value=st.session_state.mirror_camera
    )


    st.session_state.show_landmarks = st.checkbox(

        "Show Hand Landmarks",

        value=st.session_state.show_landmarks
    )


    st.divider()


    # ========================================================
    # ACCOUNT
    # ========================================================

    st.subheader(
        "👤 Account"
    )


    st.write(
        "Name:",
        st.session_state.get(
            "user_name",
            "User"
        )
    )


    st.write(
        "Email:",
        st.session_state.get(
            "user_email",
            ""
        )
    )


    st.divider()


    # ========================================================
    # SYSTEM
    # ========================================================

    st.subheader(
        "ℹ️ System Information"
    )


    st.write(
        "Computer Vision: MediaPipe"
    )

    st.write(
        "Image Processing: OpenCV"
    )

    st.write(
        "Web Interface: Streamlit"
    )

    st.write(
        "Camera Streaming: WebRTC"
    )

    st.write(
        "Robot: Virtual Simulation"
    )

    st.write(
        "Control: Hand Gesture Recognition"
    )


    st.divider()


    # ========================================================
    # RESET
    # ========================================================

    if st.button(

        "🔄 Reset All Robot Data",

        use_container_width=True
    ):

        st.session_state.robot_x = 50

        st.session_state.robot_y = 50

        st.session_state.robot_path = []

        st.session_state.history = []

        st.session_state.total_commands = 0

        st.session_state.gesture_counts = {

            "FORWARD": 0,

            "BACKWARD": 0,

            "LEFT": 0,

            "RIGHT": 0,

            "STOP": 0
        }

        st.session_state.last_recorded_command = None

        st.session_state.emergency_stop = False

        command_state.set_command(
            "STOP"
        )

        st.success(
            "Robot data reset successfully."
        )
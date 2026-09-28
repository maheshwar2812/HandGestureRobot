import streamlit as st
import re

from database import create_user, login_user


# =========================================================
# EMAIL VALIDATION
# =========================================================

def valid_email(email):

    pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"

    return re.match(pattern, email) is not None


# =========================================================
# LOGIN PAGE
# =========================================================

def login_page():

    # -----------------------------------------------------
    # PAGE TITLE
    # -----------------------------------------------------

    st.markdown(
        """
        <div style="
            text-align:center;
            margin-top:50px;
            margin-bottom:30px;
        ">

            <div style="font-size:65px;">
                🤖
            </div>

            <h1 style="color:#f8fafc;">
                Vision Robot Control
            </h1>

            <p style="
                color:#94a3b8;
                font-size:17px;
            ">
                Hand Gesture Recognition System
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # TABS
    # -----------------------------------------------------

    login_tab, create_tab = st.tabs(
        [
            "🔐 Login",
            "📝 Create Account"
        ]
    )


    # =====================================================
    # LOGIN TAB
    # =====================================================

    with login_tab:

        st.markdown("### Welcome Back 👋")

        st.write(
            "Login to access the robot control dashboard."
        )

        with st.form("login_form"):

            email = st.text_input(
                "Email Address",
                placeholder="Enter your email"
            )

            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password"
            )

            login_button = st.form_submit_button(
                "🔐 Login",
                use_container_width=True
            )


        # -------------------------------------------------
        # LOGIN BUTTON
        # -------------------------------------------------

        if login_button:

            if not email or not password:

                st.error(
                    "Please enter your email and password."
                )

            elif not valid_email(email):

                st.error(
                    "Please enter a valid email address."
                )

            else:

                user = login_user(
                    email,
                    password
                )

                if user:

                    # -------------------------------------
                    # SAVE USER SESSION
                    # -------------------------------------

                    st.session_state.logged_in = True

                    st.session_state.user_id = user[0]

                    st.session_state.user_name = user[1]

                    st.session_state.user_email = user[2]

                    st.success(
                        "Login successful!"
                    )

                    st.rerun()

                else:

                    st.error(
                        "Invalid email or password."
                    )


    # =====================================================
    # CREATE ACCOUNT TAB
    # =====================================================

    with create_tab:

        st.markdown(
            "### Create Your Account 🚀"
        )

        st.write(
            "Register to use the Vision-Based Robot Control System."
        )


        with st.form("create_account_form"):

            name = st.text_input(
                "Full Name",
                placeholder="Enter your full name"
            )

            email = st.text_input(
                "Email Address",
                placeholder="Enter your email"
            )

            password = st.text_input(
                "Password",
                type="password",
                placeholder="Minimum 6 characters"
            )

            confirm_password = st.text_input(
                "Confirm Password",
                type="password",
                placeholder="Re-enter your password"
            )

            create_button = st.form_submit_button(
                "📝 Create Account",
                use_container_width=True
            )


        # -------------------------------------------------
        # CREATE ACCOUNT BUTTON
        # -------------------------------------------------

        if create_button:

            if not name or not email or not password:

                st.error(
                    "Please fill in all fields."
                )

            elif not valid_email(email):

                st.error(
                    "Please enter a valid email address."
                )

            elif len(password) < 6:

                st.error(
                    "Password must contain at least 6 characters."
                )

            elif password != confirm_password:

                st.error(
                    "Passwords do not match."
                )

            else:

                success, message = create_user(
                    name,
                    email,
                    password
                )

                if success:

                    st.success(
                        "🎉 Account created successfully!"
                    )

                    st.info(
                        "Go to the Login tab and log in."
                    )

                else:

                    st.error(message)


# =========================================================
# LOGOUT
# =========================================================

def logout():

    st.session_state.logged_in = False

    st.session_state.user_id = None

    st.session_state.user_name = None

    st.session_state.user_email = None

    st.rerun()
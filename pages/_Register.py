"""
Registration Page for Disease Prediction System
"""
import streamlit as st

from src.auth import auth

st.set_page_config(
    page_title="Register - Disease Prediction",
    page_icon="📝",
    layout="centered"
)

# Custom CSS for centered modern design
st.markdown("""
    <style>
    .main {
        max-width: 600px;
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 50px;
        font-weight: 600;
    }
    .logo-text {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #667eea;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Main content
st.markdown('<div class="logo-text">📝 Create New Account</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Create a demo account to save your model results.</div>', unsafe_allow_html=True)


# If already logged in, show message
if auth.is_authenticated():
    st.info("✅ You are already logged in!")
    if st.button("🚪 Logout", type="primary"):
        auth.logout_user()
        st.rerun()
    st.stop()

# Registration form
with st.form("registration_form"):
    col1, col2 = st.columns(2)

    with col1:
        username = st.text_input(
            "Username *",
            placeholder="Choose a username",
            max_chars=50
        )

    with col2:
        full_name = st.text_input(
            "Full Name",
            placeholder="Your full name",
            max_chars=100
        )

    email = st.text_input(
        "Email *",
        placeholder="your.email@example.com",
        max_chars=100
    )

    password = st.text_input(
        "Password *",
        type="password",
        placeholder="Create a strong password",
        max_chars=100
    )

    confirm_password = st.text_input(
        "Confirm Password *",
        type="password",
        placeholder="Re-enter your password",
        max_chars=100
    )

    demo_only = st.checkbox("I will use synthetic records or public sample images only.")

    st.write("")
    submit_button = st.form_submit_button("✅ Create Account", use_container_width=True, type="primary")

    if submit_button:
        # Validation
        if not username or not email or not password or not confirm_password:
            st.error("⚠️ Please fill in all required fields")
        elif not demo_only:
            st.error("Confirm that you will use demo inputs only.")
        elif password != confirm_password:
            st.error("⚠️ Passwords do not match")
        else:
            with st.spinner("Creating your account..."):
                success, message = auth.register_user(
                    username=username.strip(),
                    email=email.strip().lower(),
                    password=password,
                    full_name=full_name.strip() if full_name else None
                )

                if success:
                    st.success(message)
                    st.switch_page("pages/_Login.py")
                else:
                    st.error(message)

st.write("")
st.markdown("---")

# Link to login
st.markdown("<p style='text-align: center;'>Already have an account?</p>", unsafe_allow_html=True)
if st.button("🔐 Sign In", use_container_width=True):
    st.switch_page("pages/_Login.py")

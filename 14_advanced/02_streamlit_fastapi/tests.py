import streamlit as st

st.title("AI Assistant")

name = st.text_input("Enter your name")

if name:
    st.write(f"Hello, {name}!")

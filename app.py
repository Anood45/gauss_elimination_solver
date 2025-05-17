import streamlit as st
import numpy as np
import io
import contextlib

from gauss_solver import gauss_elimination, back_substitution, verify_solution

st.set_page_config(page_title="Gaussian Elimination Solver", layout="centered")
st.title(" Gaussian Elimination Solver")
st.markdown("Enter the number of equations, then fill in the matrix values to solve the system.")

# Step 1: Choose number of equations
n = st.number_input(" Number of equations:", min_value=2, max_value=10, step=1, value=3)

# Step 2: Manual entry for A and b
st.subheader(" Coefficient matrix A:")
A = np.zeros((n, n))
for i in range(n):
    cols = st.columns(n)
    for j in range(n):
        A[i, j] = cols[j].number_input(f"A[{i+1},{j+1}]", key=f"A-{i}-{j}", value=0.0, format="%.4f")

st.subheader(" Right-hand side vector b:")
b = np.zeros(n)
for i in range(n):
    b[i] = st.number_input(f"b[{i+1}]", key=f"b-{i}", value=0.0, format="%.4f")

# Step 3: Solve the system
if st.button(" Solve"):
    st.subheader(" Gaussian Elimination Steps:")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        Ab = gauss_elimination(A.copy(), b.copy())
    st.code(buffer.getvalue())

    if Ab is not None:
        x = back_substitution(Ab)
        if x is not None:
            st.subheader(" Final Solution:")
            for i in range(n):
                st.write(f"x{i+1} = {x[i]:.4f}")
            st.session_state["last_solution"] = (A, b, x)
        else:
            st.error(" Back substitution failed: zero pivot encountered.")
    else:
        st.error(" Gaussian elimination failed: system may not have a unique solution.")

# Step 4: Verify solution (if it was solved before)
if "last_solution" in st.session_state:
    if st.button(" Verify Solution (A * x ≈ b)"):
        A, b, x = st.session_state["last_solution"]
        st.subheader(" Solution Verification:")
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            verify_solution(A, b, x)
        st.code(buffer.getvalue())

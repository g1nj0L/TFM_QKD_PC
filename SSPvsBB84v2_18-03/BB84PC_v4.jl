#=
Optimal key rates for BB84 with partial source characterization
================================================================
Implements the conic optimization from:
  "Optimal key rates for quantum key distribution with partial source characterization"

Computes the asymptotic secret key rate (Eq. 4):

    R∞ = D(G(ρ_AB) ‖ Z(G(ρ_AB))) − p_pass · λ_EC

by solving the conic optimization problem in Eq. 12 (partial characterization)
or Eq. 7 (full characterization), using the reduced Gram matrix formulation
from Appendix B (handling linearly dependent reference states).

Throughout, we use the asymptotic trick described in the paper:
    p_{Z_A} = p_{Z_B} = 1  AND  p_{X_A} = p_{X_B} = 1
This gives Tr[ρ_AB] = 2 (unphysical), but correctly recovers the asymptotic rate.
All prior probabilities become p_j = 1/2.

    ┌───────────────────────────────────────┐
====│  NOTICE: here I will implement the    │
    │  changes to code the depolarizing     │
====│  channel in the BB84 protocol.        │
    └───────────────────────────────────────┘

=#

import Hypatia      # Solver
using JuMP          # Solver things #2
using LinearAlgebra # Solver things #3
using Ket           # Solver things #4
using ConicQKD      # Solver things #5
using Plots         # Plots
using LaTeXStrings  # Plots (LaTeX rendering)
using CSV           # Plots (write and read CSV data)
using DataFrames    # Plots (data frames)

# ─────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────

const n_states = 4               # Number of Alice's settings: {0_Z, 1_Z, 0_X, 1_X}
const n_dim = 2                  # dim(span{|φⱼ⟩}) — BB84 reference states span C²
const dim_B = 3                  # Bob's system dimension: qubit {|0⟩,|1⟩} + vacuum |vac⟩
const dim_AB = n_states * dim_B  # = 12, total dimension of ρ_AB

# ─────────────────────────────────────────────────
# Reference states
# ─────────────────────────────────────────────────

@doc """
    reference_states(δ)

Construct the four BB84 reference states |φⱼ⟩ ∈ C² as columns of a 2×4 matrix.

Each state is parametrized as (see Eq. 13 in the paper):

    |φⱼ⟩ = cos(θⱼ)|0⟩ + sin(θⱼ)|1⟩

where θⱼ = κ φⱼ / 2  with  κ = 1 + δ/π  and  φⱼ ∈ {0, π, π/2, 3π/2}
for j ∈ {0_Z, 1_Z, 0_X, 1_X}. When δ = 0, these reduce to the ideal
BB84 states {|0⟩, |1⟩, |+⟩, |−⟩}.
"""
function reference_states(δ::T) where {T}
    κ = 1 + δ / π
    # Angles θⱼ = κ φⱼ / 2 for φⱼ ∈ {0, π, π/2, 3π/2}

    φ_0 = [one(T), zero(T)]                                      # |0⟩
    φ_1 = [cos(κ * π / 2),  sin(κ * π / 2)]                      # ≈ |1⟩
    φ_2 = [cos(κ * π / 4),  sin(κ * π / 4)]                      # ≈ |+⟩
    φ_3 = [cos(3κ * π / 4), sin(3κ * π / 4)]                     # ≈ |−⟩
    return hcat(φ_0, φ_1, φ_2, φ_3)
end

# ─────────────────────────────────────────────────
# Channel model: source → loss → dark counts → measurement
# ─────────────────────────────────────────────────

@doc """
    source_replacement_state(δ)

Build the source replacement state |Ψ⟩⟨Ψ|_{AA'} (Eq. 2 in the paper):

    |Ψ⟩ = Σⱼ √pⱼ |j⟩_A |φⱼ⟩_{A'}

where A is Alice's 4-dimensional ancilla and A' is the emitted qubit (C²).
With the asymptotic trick, pⱼ = 1/2 for all j.

Returns a (n_states · n_dim) × (n_states · n_dim) = 8 × 8 density matrix.
"""
function source_replacement_state(δ::T) where {T}
    φ = reference_states(δ)
    ψ = zeros(T, n_states * n_dim)
    for j in 1:n_states
        eⱼ = zeros(T, n_states); eⱼ[j] = 1
        ψ .+= sqrt(T(1) / 2) * kron(eⱼ, φ[:, j])
    end
    return ψ * ψ'
end

# [CHANGE 18/03/26]
# This is the new function where I implement the depolarizing channel.

@doc """
    apply_depolarizing_channel(ρ_AA′, p_dep)

The polarizing channel, Ωₚ, can be defined as the channel where depolarizing happens with probability p and nothing happens with probability 1-p:

        Ωₚ(ρ) = p · Ω(ρ) + (1-p) · ρ

where

        Ω(ρ) = tr(ρ) · I/2

and I is the 2×2 identity matrix (we are working within the qubit space, therefore the 1/2 term in the identity aswell).

[TO-DO] EXPLAIN THIS FURTHER AND BE SURE IT'S WELL IMPLEMENTED!!

[DOUBT 18/03/26] One can represent this channel in terms of its (non-unique) Kraus operators, i.e., its Kraus representation. One such option is the following one:

        A₀ = √[1-p] · I
        A₁ = √[p/3] · σ_x
        A₂ = √[p/3] · σ_y
        A₃ = √[p/3] · σ_z
        
which can be proven to recover Ωₚ(ρ)

        Ωₚ(ρ) = A₀ ρ A₀† + A₁ ρ A₁† + A₂ ρ A₂† + A₃ ρ A₃†

> I'm not sure if this is the most intelligent way to implement this, since we are acting on a small system (qubit space) and the depolarizing operation is relatively trivial to compute. Therefore, an adhoc solution could be more computationally efficient, even though there might be some problems associated with this solution I'm not aware of. I'll try to code it this way (what is written before this DOUBT) and in case I get strange results, I'll do the Kraus' operators approach.
"""
function apply_depolarizing_channel(ρ_AA′::AbstractMatrix, p_dep::T) where {T}
    # When the depolarization occurs, the state of the subsystem A′ (the one
    # who travels through the channel from Alice to Bob) gets transformed into
    # the maximally entangled state, I/2 (because n_dim=2, since we are working
    # with qubits). In either case (depolarization or not), the subsystem
    # Alice keeps, A, remains untouched, as it doesn't travel through the channel.
    
    I_A′ = Matrix{T}(I, n_dim, n_dim)
    ρ_A = partial_trace(ρ_AA′, 2, [n_states, n_dim])

    #             DEPOLARIZATION         NO-DEPOLARIZATION
    #     ┌───────────────────────────┐ ┌─────────────────┐
    return p_dep * kron(ρ_A, I_A′ / 2) + (1-p_dep) * ρ_AA′
end

@doc """
    apply_loss_channel(ρ_AA′, η)

Apply the lossy channel to Alice's emitted photon, mapping A' (C²) → B (C³).

With probability η the photon arrives intact in Bob's qubit subspace,
with probability 1−η the photon is lost and Bob gets the vacuum state:

    ρ_AB = η (I_A ⊗ V) ρ_{AA'} (I_A ⊗ V)† + (1−η) ρ_A ⊗ |vac⟩⟨vac|

where V: C² → C³ embeds the qubit into Bob's space {|0⟩, |1⟩, |vac⟩},
and ρ_A = Tr_{A'}[ρ_{AA'}].

Returns a dim_AB × dim_AB = 12 × 12 density matrix ρ_AB.
"""
function apply_loss_channel(ρ_AA′::Matrix{T}, η::T) where {T}
    # V: C² → C³ embeds qubit into Bob's space {|0⟩, |1⟩, |vac⟩}
    V = zeros(T, dim_B, n_dim)
    V[1, 1] = 1    # |0⟩ → |0⟩_B
    V[2, 2] = 1    # |1⟩ → |1⟩_B

    vac = zeros(T, dim_B); vac[dim_B] = 1
    I_A = Matrix{T}(I, n_states, n_states)

    ρ_A = partial_trace(ρ_AA′, 2, [n_states, n_dim])

    # Photon arrives (prob η) or is lost (prob 1−η)
    IV = kron(I_A, V)
    return η * IV * ρ_AA′ * IV' + (1 - η) * kron(ρ_A, vac * vac')
end

@doc """
    apply_dark_count_channel(ρ_AB, d)

Apply the dark count noise channel to Bob's system (C³).

Implements Eq. 10 from arXiv:2503.06328v2 by Nahar, Tupkary & Lütkenhaus (2025):

    Φ_d(ρ) = ⟨vac|ρ|vac⟩ · σ_vac + (1−d) Π ρ Π + Tr[Π ρ Π] · d Π/2

where:
  • Π = |0⟩⟨0| + |1⟩⟨1| is the projector to the qubit subspace
  • σ_vac = (1−d)² |vac⟩⟨vac| + d(1−d/2) Π

The three terms have a clear physical interpretation:
  Term 1: vacuum input → stays vacuum (no dark count) or random click (dark count)
  Term 2: qubit passes through coherently (no dark count on other detector)
  Term 3: qubit depolarized (dark count on other detector → random assignment)

Assumes both detectors have the same dark count rate d (required
for the simple qubit squasher).

The channel acts as I_A ⊗ Φ_d on the bipartite state ρ_AB.
"""
function apply_dark_count_channel(ρ_AB::Matrix{T}, d::T) where {T}
    # Projectors on Bob's C³
    vac = zeros(T, dim_B); vac[dim_B] = 1
    Π = zeros(T, dim_B, dim_B); Π[1, 1] = 1; Π[2, 2] = 1

    # What happens to vacuum: stays vacuum or dark count → random click
    σ_vac = (1 - d)^2 * vac * vac' + d * (1 - d / 2) * Π

    # Bipartite operators
    I_A = Matrix{T}(I, n_states, n_states)
    P_vac = kron(I_A, vac')       # projects Bob onto |vac⟩ (shrinks dim)
    P_qubit = kron(I_A, Π)        # projects Bob onto qubit subspace

    # Term 1: vacuum component → σ_vac
    ρ_A_vac = P_vac * ρ_AB * P_vac'       # n_states × n_states: Alice's state when Bob had vacuum
    term1 = kron(ρ_A_vac, σ_vac)

    # Term 2: qubit passes through coherently (prob 1−d)
    ρ_qubit = P_qubit * ρ_AB * P_qubit
    term2 = (1 - d) * ρ_qubit

    # Term 3: qubit depolarized by dark count (prob d)
    ρ_A_qubit = partial_trace(ρ_qubit, 2, [n_states, dim_B])
    term3 = d * kron(ρ_A_qubit, Π / 2)

    return term1 + term2 + term3
end

# ─────────────────────────────────────────────────
# Observation constraints: Tr[(|j⟩⟨j|_A ⊗ Γ_k) ρ_AB]
# ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────
# Channel yields: full physical simulation
# ─────────────────────────────────────────────────

# [CHANGE 18/03/26]
# Here we implement the changes to make the function 'channel_yields' call
# the new 'apply_depolarizing_channel' function.

@doc """
    channel_yields(η, δ, p_d)

Compute the BB84 yields Y_{k|j} by physically simulating the channel:

    1. Build the source replacement state |Ψ⟩⟨Ψ|_{AA'}
    2. Apply the depolarizing channel (probability p_dep)
    3. Apply the loss channel (transmittance η) to map A' → B
    4. Apply the dark count noise channel (Eq. 10 of Nahar et al.)
    5. Measure with Bob's ideal POVM via observed_statistics()

Returns a n_states × n_states matrix where entry [j, k] gives Y_{k|j}.

This is equivalent to the analytical yield formula (Eq. D9 of the paper)
up to O(p_d²) terms — the noise channel retains the exact (1 − p_d/2)
factor rather than approximating it as 1.
"""
function channel_yields(η::T, δ::T, p_d::T, p_dep::T) where {T}
    # [CHANGE 18/03/26]
    # Add the depolarization probability 'p_dep' to 'channel_yields'.

    ρ_AA′ = source_replacement_state(δ)
    ρ_AA′ = apply_depolarizing_channel(ρ_AA′, p_dep)
    ρ_AB = apply_loss_channel(ρ_AA′, η)
    ρ_AB = apply_dark_count_channel(ρ_AB, p_d)

    p = fill(T(1) / 2, n_states)
    pⱼ_Ykj = observed_statistics(ρ_AB)  # Maybe change to directly return this?? IDK 
    return Diagonal(inv.(p)) * pⱼ_Ykj
end

# ─────────────────────────────────────────────────
# Channel model (Appendix D.1) (no longer used)
# ─────────────────────────────────────────────────

@doc """
    channel_yields_old(η, δ, p_d)

Compute the BB84 yields Y_{k|j} from the channel model (Eq. D9 in the paper). (no longer used in this version of the code)

Returns a n_states × n_states matrix where entry [j, k] gives Y_{k|j}, the probability
that Bob obtains outcome k given Alice selected setting j.

Rows j:    Alice's settings {0_Z, 1_Z, 0_X, 1_X}
Columns k: Bob's outcomes   {Z0,  Z1,  X0,  X1}

Uses the asymptotic trick p_{Z_B} = p_{X_B} = 1.
Dark-count terms are kept to O(p_d), neglecting O(p_d²).
"""
function channel_yields_old(η::T, δ::T, p_d::T) where {T}
    κ = 1 + δ / π
    # The angle 2θⱼ = κ φⱼ appears in the yield formulas
    κφ = (T(0), κ * π, κ * π / 2, 3 * κ * π / 2)

    # Initialize with dark-count-only contribution (no photon arrives)
    Y = fill((1 - η) * p_d * (1 - p_d / 2), n_states, n_states)

    for j in 1:n_states
        c = cos(κφ[j])
        s = sin(κφ[j])
        # Z-basis yields (Eq. D9, first two lines)
        Y[j, 1] += η / 2 * (1 + (1 - p_d) * c)   # Y_{Z0|j}
        Y[j, 2] += η / 2 * (1 - (1 - p_d) * c)   # Y_{Z1|j}
        # X-basis yields (Eq. D9, last two lines)
        Y[j, 3] += η / 2 * (1 + (1 - p_d) * s)   # Y_{X0|j}
        Y[j, 4] += η / 2 * (1 - (1 - p_d) * s)   # Y_{X1|j}
    end
    return Y
end

# [CHANGE 18/03/26]
# Add the depolarization probability 'p_dep' to the parameters (needed for)
# the function 'channel_yields'.

@doc """
    Z_basis_error_rate(η, δ, p_d)

Compute the Z-basis bit error rate e_Z (Eq. D10):

    e_Z = (Y_{Z1|0} + Y_{Z0|1}) / (Y_{Z0|0} + Y_{Z1|0} + Y_{Z0|1} + Y_{Z1|1})
"""
function Z_basis_error_rate(η::T, δ::T, p_d::T, p_dep::T) where {T}
    # [CHANGE 18/03/26]
    # Add the depolarization probability 'p_dep' to 'channel_yields'.

    Y = channel_yields(η, δ, p_d, p_dep)
    return (Y[1, 2] + Y[2, 1]) / sum(Y[1:2, 1:2])
end

# ─────────────────────────────────────────────────
# Observation constraints: Tr[(|j⟩⟨j|_A ⊗ Γ_k) ρ_AB]
# ─────────────────────────────────────────────────

@doc """
    observed_statistics(ρ_AB)

Compute the joint probabilities Tr[(|j⟩⟨j|_A ⊗ Γ_k) ρ_AB] for all Alice
settings j and Bob outcomes k. Returns a n_states × n_states matrix of p_j · Y_{k|j}.

Bob's POVM elements (Eq. D3) are embedded in C³ = span{|0⟩, |1⟩, |vac⟩}:
    Γ_{Z0} = |0⟩⟨0|,  Γ_{Z1} = |1⟩⟨1|,  Γ_{X0} = |+⟩⟨+|,  Γ_{X1} = |−⟩⟨−|
with p_{Z_B} = p_{X_B} = 1 (asymptotic trick).
"""
function observed_statistics(ρ_AB::AbstractMatrix)
    # Alice's projectors |j⟩⟨j| in C^n_states
    Π_A = [proj(j, n_states) for j in 1:n_states]

    # Bob's POVM elements embedded in C³
    # Z basis: |0⟩⟨0|, |1⟩⟨1| in the qubit subspace
    Γ_0Z = zeros(dim_B, dim_B); Γ_0Z[1, 1] = 1
    Γ_1Z = zeros(dim_B, dim_B); Γ_1Z[2, 2] = 1
    # X basis: |±⟩⟨±| in the qubit subspace
    Γ_0X = zeros(dim_B, dim_B); Γ_0X[1:2, 1:2] = [1 1; 1 1] / 2
    Γ_1X = zeros(dim_B, dim_B); Γ_1X[1:2, 1:2] = [1 -1; -1 1] / 2

    Γ_B = [Γ_0Z, Γ_1Z, Γ_0X, Γ_1X]

    # Compute Tr[(|j⟩⟨j| ⊗ Γ_k) ρ_AB] for all j, k
    # dot(A, B) = Tr(A†B) = Tr(ρ_AB · (|j⟩⟨j| ⊗ Γ_k))  since ρ_AB is Hermitian

    return real.([dot(ρ_AB, kron(Π_A[j], Γ_B[k])) for j in 1:n_states, k in 1:n_states]) 
end

# ─────────────────────────────────────────────────
# Sifting and key maps Ĝ, Ẑ (after facial reduction)
# ─────────────────────────────────────────────────

@doc """
    sifting_kraus()

Kraus operators for the facially reduced sifting map Ĝ (derived from Eq. D4–D5).

This map selects rounds used for key generation: Alice chose Z basis and Bob
got a conclusive qubit outcome. After facial reduction, it maps
C^{n_states ⊗ dim_B} = C¹² → C^{n_dim ⊗ 2} = C⁴ by:
  • Projecting Alice from C⁴ onto the Z subspace {|0⟩, |1⟩} (→ C²)
  • Projecting Bob from C³ onto the qubit subspace {|0⟩, |1⟩} (→ C²)

Returns a single Kraus operator (4 × 12 matrix).
"""
function sifting_kraus()
    project_A = Matrix(I, n_dim, n_states)    # C⁴ → C²: keeps |0⟩, |1⟩
    project_B = Matrix(I, 2, dim_B)           # C³ → C²: drops |vac⟩
    return [kron(project_A, project_B)]
end

@doc """
    key_map_kraus()

Kraus operators for the key map Ẑ (pinching channel in Alice's Z basis).

Alice assigns bit 0 when j = 0_Z and bit 1 when j = 1_Z.
Acts on the 4D output space of Ĝ (C² ⊗ C²):
    Z₀ = |0⟩⟨0|_A ⊗ I_B
    Z₁ = |1⟩⟨1|_A ⊗ I_B

Returns two Kraus operators (each 4 × 4).
"""
function key_map_kraus()
    Z₀ = kron(proj(1, n_dim), I(2))   # Alice bit = 0
    Z₁ = kron(proj(2, n_dim), I(2))   # Alice bit = 1
    return [Z₀, Z₁]
end

# ─────────────────────────────────────────────────
# Error correction cost
# ─────────────────────────────────────────────────

# [CHANGE 18/03/26]
# Add the depolarization probability 'p_dep' to the parameters (needed for)
# the function 'channel_yields'.

@doc """
    error_correction_cost(η, δ, p_d)

Compute H(A|B) for the Z basis from the channel model, i.e., the conditional
entropy of Alice's bit given Bob's measurement in the Z basis.

This is the error correction cost per sifted key bit before applying the
inefficiency factor f (λ_EC = f · h(e_Z)).
"""
function error_correction_cost(η::T, δ::T, p_d::T, p_dep::T) where {T}
    # [CHANGE 18/03/26]
    # Add the depolarization probability 'p_dep' to 'channel_yields'.

    Y = channel_yields(η, δ, p_d, p_dep)
    # Joint distribution P(A=a, B=b) for Z basis:
    # P(A=a, B=b) = (1/2) · Y_{Zb|Za} since P(A=a) = 1/2
    # Remember that the 1/2 comes from the fact that we chose between bit 0 and 1 randomnly
    P_ZZ = Y[1:2, 1:2] / 2
    return conditional_entropy(P_ZZ)
end

# ─────────────────────────────────────────────────
# Main optimization: Eve's information
# ─────────────────────────────────────────────────

@doc """
    optimize_eve_information(ϵ, η, δ, p_d)

Solve the conic optimization problem to find the minimum of
D(G(ρ_AB) ‖ Z(G(ρ_AB))), i.e., the conditional entropy of Alice's key
from Eve's perspective.

When ϵ = 0: solves the fully characterized problem (Eq. 7)
             with facial reduction on ρ_AB.
When ϵ > 0: solves the partially characterized problem (Eq. 12)
             using the reduced Gram matrix from Appendix B.

Arguments:
  ϵ   — source deviation bound: ⟨φⱼ|ρⱼ|φⱼ⟩ ≥ 1 − ϵ (Eq. 1)
  η   — overall channel transmittance (η = η_c · η_d)
  δ   — state preparation flaw magnitude
  p_d — dark count probability

Returns the optimal value of D(G(ρ_AB) ‖ Z(G(ρ_AB))) in bits.
"""
function optimize_eve_information(ϵ::T, η::T, δ::T, p_d::T, p_dep::T) where {T<:Real}
    model = GenericModel{T}(() -> Hypatia.Optimizer{T}(; verbose = false)) # Change "false" to "true" if needed for debugging purposes
    psd_cone, wrapper, hermitian_space = Ket._sdp_parameters(false)  # real-valued

    # ── Reference states and fidelity matrix ──
    #
    # φ is a 2 × 4 matrix whose columns are the reference states |φⱼ⟩ ∈ C²
    # F[i,j] = ⟨φᵢ|φⱼ⟩ is the known fidelity (Gram) matrix of the reference states
    φ = reference_states(δ)
    F = φ' * φ

    # ── Prior probabilities (asymptotic trick) ──
    #
    # With p_{Z_A} = p_{X_A} = 1, all four pⱼ = 1/2 (Eq. D1)
    # The normalization factor 1/√(pᵢpⱼ) = 2 for all i,j
    p = fill(T(1) / 2, n_states)

    # ═══════════════════════════════════════════
    # Branch: Full characterization (ϵ = 0)
    # Conic problem from Eq. 8
    # ═══════════════════════════════════════════
    if ϵ == 0
        # Since the reference states |φⱼ⟩ span only C², the marginal ρ_A has
        # rank ≤ 2. We apply facial reduction: ρ_AB = V ρ_core V†, where
        # V projects onto the support of ρ_A ⊗ I_B.
        _, _, v = svd(φ)                                    # v is 4 × 2
        V = kron(v, I(dim_B))                               # 12 × 6 isometry
        @variable(model, ρ_core[1:n_dim*dim_B, 1:n_dim*dim_B] ∈ hermitian_space)
        ρ_AB = wrapper(V * ρ_core * V')

        # Source constraint (Eq. 7, first line):
        #   ⟨i|ρ_A|j⟩ / √(pᵢp_j) = ⟨ψⱼ|ψᵢ⟩ = ⟨φⱼ|φᵢ⟩ = F[i,j]
        # Off-diagonal only (j > i); diagonal handled separately below
        ρ_A = partial_trace(ρ_AB, 2, [n_states, dim_B])
        for j in 2:n_states, i in 1:j-1
            @constraint(model, 2 * ρ_A[j, i] == F[i, j])
        end

    # ═══════════════════════════════════════════
    # Branch: Partial characterization (ϵ > 0)
    # Conic problem from Eq. 12, using Appendix B reduced Gram matrix
    # ═══════════════════════════════════════════
    else

        # No facial reduction: the actual states are unknown, so ρ_AB
        # may have support anywhere in C^{n_states} ⊗ C^{dim_B} = C¹²
        @variable(model, ρ_AB[1:dim_AB, 1:dim_AB] ∈ hermitian_space)
        ρ_A = partial_trace(ρ_AB, 2, [n_states, dim_B])

        # ── Reduced Gram matrix (Appendix B) ──
        #
        # The reference states |φⱼ⟩ are linearly dependent (4 states in C²),
        # so the full (2n_states × 2n_states) Gram matrix from Eq. 11 cannot have full rank.
        # Following Appendix B, we instead use a basis {|l⟩}_{l=0}^{n_dim-1} for
        # span{|φⱼ⟩} (here, the computational basis {|0⟩, |1⟩} of C²).
        #
        # G_reduced is the (n_dim + n_states) × (n_dim + n_states) = 6 × 6 Gram matrix:
        #
        #         ┌────────────────┬──────────────────────┐
        #         │  ⟨l|l'⟩         │  ⟨l|φⱼ⊥⟩             │
        #         │  (= I_{n_dim}) │                      │
        #         ├────────────────┼──────────────────────┤
        #         │  ⟨φᵢ⊥|l⟩        │   ⟨φᵢ⊥|φⱼ⊥⟩          │
        #         │                │                      │
        #         └────────────────┴──────────────────────┘
        #   indices: 1:n_dim          n_dim+1 : n_dim+n_states

        dim_G = n_dim + n_states   # = 6
        @variable(model, G_reduced[1:dim_G, 1:dim_G] ∈ psd_cone)

        # (Appendix B) All states normalized: G_{m,m} = 1 for all m
        @constraint(model, diag(G_reduced) .== 1)

        # (Appendix B) Basis orthonormality: G_{l,l'} = 0 for l ≠ l'
        # With n_dim = 2, there is only one such constraint:
        @constraint(model, G_reduced[1, 2] == 0)

        # ── Derived inner products ⟨φᵢ|φⱼ⊥⟩ (Appendix B, Eq. B4) ──
        #
        # Since |φᵢ⟩ = Σₗ cₗ⁽ⁱ⁾|l⟩ (Eq. B1), where cₗ⁽ⁱ⁾ = φ[l, i]:
        #   ⟨φᵢ|φⱼ⊥⟩ = Σₗ c̄ₗ⁽ⁱ⁾ ⟨l|φⱼ⊥⟩ = Σₗ c̄ₗ⁽ⁱ⁾ G_reduced[l, n_dim + j]
        #
        # This is computed as: dot(φ[:, i], G_reduced[1:n_dim, n_dim + j])

        φᵢ_φⱼ⊥ = [dot(φ[:, i], G_reduced[1:n_dim, n_dim+j]) for i in 1:n_states, j in 1:n_states]

        # (Appendix B, Eq. B5) Orthogonality: ⟨φⱼ|φⱼ⊥⟩ = 0 for all j
        @constraint(model, diag(φᵢ_φⱼ⊥) .== 0)

        # ── Assemble full (2n_states × 2n_states) Gram matrix G (Eq. 11) ──
        #
        # This is derived from G_reduced; it is NOT a separate variable.
        #
        #   G = ┌──────────┬──────────────┐
        #       │ F        │  φᵢ_φⱼ⊥      │   F[i,j] = ⟨φᵢ|φⱼ⟩ (known)
        #       ├──────────┼──────────────┤
        #       │ φᵢ⊥_φⱼ   │  ⟨φᵢ⊥|φⱼ⊥⟩   │   from G_reduced lower-right block
        #       └──────────┴──────────────┘

        φᵢ⊥_φⱼ⊥ = G_reduced[n_dim+1:end, n_dim+1:end]
        G = wrapper([F φᵢ_φⱼ⊥; φᵢ_φⱼ⊥' φᵢ⊥_φⱼ⊥])

        # ── Source constraint (Eq. 12, first line) ──
        #
        # For pure states |ψⱼ⟩ = √(1−ϵ)|φⱼ⟩ + √ϵ|φⱼ⊥⟩ (Eq. 7):
        #
        #   ⟨i|ρ_A|j⟩        ⟨ψⱼ|ψᵢ⟩
        #   ───────── = ──────────────────────────────────────────────────
        #   √(pᵢ p_j)   (1−ϵ)⟨φⱼ|φᵢ⟩ + √(ϵ(1−ϵ))(⟨φⱼ⊥|φᵢ⟩ + ⟨φⱼ|φᵢ⊥⟩)
        #                 + ϵ ⟨φⱼ⊥|φᵢ⊥⟩
        #
        # Off-diagonal only (j > i); diagonal handled separately below

        for j in 2:n_states, i in 1:j-1
            rhs = (1 - ϵ) * F[i, j] +
                  √(ϵ * (1 - ϵ)) * (G[i, j+n_states] + G[i+n_states, j]) +
                  ϵ * G[i+n_states, j+n_states]
            @constraint(model, 2 * ρ_A[j, i] == rhs)
        end
    end

    # ═══════════════════════════════════════════
    # Constraints common to both branches
    # ═══════════════════════════════════════════

    # ── Diagonal of ρ_A (source constraint, i = j case) ──
    #
    # ⟨j|ρ_A|j⟩ = p_j = 1/2
    # This is the i = j case of the source constraint, which simplifies
    # because (1−ϵ) + ϵ = 1 and ⟨φⱼ⊥|φⱼ⟩ = 0.
    # Also implicitly sets Tr[ρ_AB] = Σⱼ p_j = 2 (asymptotic trick).
    @constraint(model, diag(ρ_A) .== p)

    # ── Yield constraints (Eq. 12, second line) ──
    #
    # Tr[(|j⟩⟨j|_A ⊗ Γ_k) ρ_AB] = p_j Y_{k|j}
    #
    # We constrain Y_{k|j} for k ∈ {Z0, Z1, X0} (columns 1–3).
    # Column 4 (X1) is redundant given diag(ρ_A) = p, since
    # Σ_k Tr[(|j⟩⟨j| ⊗ Γ_k) ρ_AB] = Tr[(|j⟩⟨j| ⊗ I) ρ_AB] = p_j.
    # Including it would create a linearly dependent constraint, causing
    # numerical issues for the interior-point solver.

    # [CHANGE 18/03/26]
    # Add the depolarization probability 'p_dep' to 'channel_yields'.

    Y_target = channel_yields(η, δ, p_d, p_dep)
    p_j_Ykj = observed_statistics(ρ_AB)                  # p_j · Y_{k|j}
    Y_model = Diagonal(inv.(p)) * p_j_Ykj                # Y_{k|j}
    @constraint(model, Y_model[:, 1:3] .== Y_target[:, 1:3])

    # ── QKD cone constraint (Eq. 8) ──
    #
    # (h, ρ_AB) ∈ K_QKD  ⟺  ρ_AB ≥ 0 and h ≥ −H(Ĝ(ρ_AB)) + H(Ẑ(Ĝ(ρ_AB)))
    #
    # where Ĝ, Ẑ are the facially reduced sifting and key maps.

    Ĝ_kraus = sifting_kraus()
    Ẑ_kraus = key_map_kraus()
    ẐĜ_kraus = vec([Ẑk * Ĝk for Ĝk in Ĝ_kraus, Ẑk in Ẑ_kraus])

    ρ_vec = svec(ρ_AB)
    @variable(model, h)
    @objective(model, Min, h / log(T(2)))    # nats → bits
    @constraint(model, [h; ρ_vec] in EpiQKDTriCone{T,T}(Ĝ_kraus, ẐĜ_kraus, 1 + length(ρ_vec)))

    # ── Solve ──
    optimize!(model)
    return objective_value(model)
end

# ─────────────────────────────────────────────────
# Secret key rate (Eq. 4)
# ─────────────────────────────────────────────────

# [CHANGE]
# Add the probability of depolarization 'p_dep' as a parameter.
#   > p_dep as a variable
#   > p_dep as a parameter in 'optimize_eve_information' and 'error_correction_cost'.

@doc """
    secret_key_rate(ϵ, δ, D)

Compute the asymptotic secret key rate R∞ (Eq. 4) for BB84 with partial
source characterization:

    R∞ = D(G(ρ_AB) ‖ Z(G(ρ_AB))) − f · H(A|B)

Arguments:
  ϵ — source deviation bound (Eq. 1): ⟨φⱼ|ρⱼ|φⱼ⟩ ≥ 1 − ϵ
  δ — state preparation flaw magnitude (δ ∈ [0, π))
  D — channel distance in km

Channel parameters (Appendix D.1):
  α   = 0.2 dB/km   (fiber loss)
  η_d = 0.73         (detector efficiency)
  f   = 1.16         (error correction inefficiency)
  p_d = 10⁻⁶         (dark count probability)
"""
function secret_key_rate(ϵ::T, δ::T, D) where {T<:Real}
    # CHANNEL PARAMETERS
    α = T(2) / 10               # fiber loss (dB/km)
    η_d = T(1)                  # detector efficiency (comparision with the paper)
    #η_d = T(73) / 100          # detector efficiency (correct value)
    η = η_d * 10^(-α * D / 10)  # total transmittance: η_c · η_d
    f = T(116) / 100            # error correction inefficiency
    p_d = inv(T(10^6))          # dark count probability
    p_dep = T(1) / 10           # depolarization probability

    # R∞ = D(G(ρ) ‖ Z(G(ρ)))  −  f · H(A|B)
    #     └─── Eve's info ──┘  └─ EC cost ─┘
    eve_info = optimize_eve_information(ϵ, η, δ, p_d, p_dep)
    ec_cost = f * error_correction_cost(η, δ, p_d, p_dep)

    return eve_info - ec_cost
end

#secret_key_rate(1e-6, 0.064, 0) # For debugging purposes

# ─────────────────────────────────────────────────
# Graphical representations
# ─────────────────────────────────────────────────

# [ GRAPH CALCULATIONS ]

ϵ_mod = [0.0, 1e-6, 1e-3] # The three values seen in the paper
δ_mod = 0.0               # Variable that quantifies the errors in the characterization of Alice's font
D_mod = 0:1:255           # From 0 to 255 km (1km at a time)

# Dictionary to save the SKR of each ϵ:
#   The dictionary allows us to save information accessible through tags
#   (just as a regular dictionary: if we search a word we get a definition).
#   In this case: results[ϵ_value]=[D_values_vector]
results = Dict{Float64, Vector{Float64}}()

for ϵ in ϵ_mod # With this loop we go through all the values of ϵ
    println("──────| Line for ϵ = $ϵ |──────")
    skr_array = Float64[] # Empty array for the lines
    
    for D in D_mod # With this loop we go trough all the distances D
        println("  > Distance D = $D km")

        rate = secret_key_rate(ϵ, δ_mod, D) # Here we compute the SKR each time

        # If the SKR drops below zero, we no longer can extract any key securely
        if rate <= 1e-15
            println("    >> SKR = $rate. If we go further, it will be negative aswell, so we can truncate the loop.") 
            zeros_left = length(D_mod) - length(skr_array) # What is missing to fill the vector for each distance
            append!(skr_array, zeros(zeros_left)) # Fill skr_array with zeros
            break # Eixit the loop
        else
            println("    >> SKR = $rate")
            
            # Save the value of rate (specific round) at the end of skr_array
            push!(skr_array, rate) # We save the positive values
        end
    end
    
    # Save the results of the specific ϵ inside the dictionary
    results[ϵ] = skr_array
end

# [ SAVE THE DATA ]

# Initialize a DataFrame with the column of the distances
df_export = DataFrame(D_km = D_mod)

# Add a column for each ϵ of the dictionary
for ϵ in sort(collect(keys(results))) # We sort it from smaller values to bigger ones (it's not needed, just for readability purposes)
    column_name = "SKR_$ϵ" # Name of the column
    
    df_export[!, column_name] = results[ϵ] # Insert the complete array from the dictionary as a new column
end

# Save the results into a CSV
CSV.write("SSPvsBB84v2_18-03/data/BB84_depolarized_data.csv", df_export)
println()
println("BB84 data successfully saved into 'BB84_depolarized_data.csv'!")
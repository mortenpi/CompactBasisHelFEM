module CompactBasisHelFEM
using ContinuumArrays: SimplifyStyle
using CompactBases: CompactBases, Basis, @materialize, LazyArrays, ContinuumArrays,
    QuasiAdjoint, QuasiDiagonal, Derivative, BroadcastQuasiArray
using HelFEM: HelFEM

# This comes from EllipsisNotation via CompactBases, but can't be imported with using
const .. = CompactBases.:(..)

struct HelFEMBasis <: Basis{Float64}
    b :: HelFEM.RadialBasis
end

function Base.axes(b::HelFEMBasis)
    rmin, rmax = first(HelFEM.boundaries(b.b)), last(HelFEM.boundaries(b.b))
    (CompactBases.Inclusion(rmin..rmax), Base.OneTo(length(b.b)))
end

function Base.getindex(b::HelFEMBasis, r::Number, i)
    r ∈ axes(b,1) && i ∈ axes(b,2) || throw(BoundsError(b, [r,i]))
    b.b([r])[i]
end

function Base.getindex(b::HelFEMBasis, r::Vector, i)
    all(∈(axes(b,1)), r) && (i isa Colon || all(∈(axes(b,2)),i)) || throw(BoundsError(b, [r,i]))
    b.b(r)[:,i]
end

Base.getindex(b::HelFEMBasis, r::AbstractVector, i) = b[collect(r), i]

# Overlap matrix
@materialize function *(Ac::QuasiAdjoint{<:Any,<:HelFEMBasis}, B::HelFEMBasis)
    SimplifyStyle
    T -> begin
        A = parent(Ac)
        Matrix{T}(undef, length(A.b), length(B.b))
    end
    dest::Matrix{T} -> begin
        A = parent(Ac)
        dest .= HelFEM.radial_integral(A.b, 0, B.b)
    end
end

# Potential
@materialize function *(Ac::QuasiAdjoint{<:Any,<:HelFEMBasis}, D::QuasiDiagonal, B::HelFEMBasis)
    SimplifyStyle
    T -> begin
        A = parent(Ac)
        Matrix{T}(undef, length(A.b), length(B.b))
    end
    dest::Matrix{T} -> begin
        A = parent(Ac)
        rs = HelFEM.quadraturepoints(A.b)
        @assert rs == HelFEM.quadraturepoints(B.b)
        ws = HelFEM.quadratureweights(A.b)
        ys = getindex.(Ref(D.diag), rs)
        A_bf, B_bf = HelFEM.basisvalues(A.b), HelFEM.basisvalues(B.b)
        dest .= A_bf' * (rs .* rs .* ys .* ws .* B_bf)
    end
end


@materialize function *(Ac::QuasiAdjoint{<:Any,<:HelFEMBasis}, D::Derivative, B::HelFEMBasis)
    SimplifyStyle
    T -> begin
        A = parent(Ac)
        Matrix{T}(undef, length(A.b), length(B.b))
    end
    dest::Matrix{T} -> begin
        A = parent(Ac)
        dest .= HelFEM.radial_integral(A.b, 0, B.b, rderivative = true)
    end
end

@materialize function *(Ac::QuasiAdjoint{<:Any,<:HelFEMBasis}, Dc::QuasiAdjoint{<:Any,<:Derivative}, D::Derivative, B::HelFEMBasis)
    SimplifyStyle
    T -> begin
        A = parent(Ac)
        Matrix{T}(undef, length(A.b), length(B.b))
    end
    dest::Matrix{T} -> begin
        A = parent(Ac)
        dest .= HelFEM.radial_integral(A.b, 0, B.b, lderivative = true, rderivative = true)
    end
end

function Sinvh(A::HelFEMBasis)
    _, Sinvh = HelFEM.overlap(A.b, invh=true)
    return Sinvh
end

# Interpolating functions over HelFEMBasis
function Base.:(\ )(B::HelFEMBasis, f::BroadcastQuasiArray)
    @assert B.b.primbas == 4 # only works for LIPs at the moment
    axes(f,1) == axes(B,1) || throw(DimensionMismatch("Function on $(axes(f,1).domain) cannot be interpolated over basis on $(axes(B,1).domain)"))
    cs = HelFEM.controlpoints(B.b)
    collect(cs[2:end-1] .* f[cs[2:end-1]])
end

end

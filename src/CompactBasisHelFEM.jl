module CompactBasisHelFEM
using ContinuumArrays: SimplifyStyle
using CompactBases: CompactBases, Basis, @materialize, LazyArrays, ContinuumArrays,
    QuasiAdjoint, QuasiDiagonal, Derivative
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

end

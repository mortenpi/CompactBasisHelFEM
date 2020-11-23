module CompactBasisHelFEM
using HelFEM: RadialBasis
using CompactBases: Basis

struct HelFEMBasis{T} <: Basis{T}
    b :: RadialBasis
end

end

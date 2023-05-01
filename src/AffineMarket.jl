module AffineMarket
export MarketDynamics,MarketState,Observations,ObservationsFromState!,UpdateState!,ExpectedObservations!
#include("VectorStoredArray.jl")
using VectorStoredArray: StoredNode, Storage, StoredValue
FloatType=Float64

struct MarketDynamics
    NF::Int64
    Initialrate::FloatType
    k1::FloatType
    theta::FloatType
    s1::FloatType
    s0::FloatType
    b0::FloatType
    thetaQ::FloatType
    b::FloatType
    se2::FloatType
end

struct OneCurrency <: StoredNode
    X0::StoredValue
    X::StoredValue
end

struct MarketState <: StoredNode
    USD::OneCurrency
end

struct OneCurrencyObs <: StoredNode
    X0::StoredValue
    X::StoredValue
    ON::StoredValue
end

struct Observations <: StoredNode
    USD::OneCurrencyObs
end

function MarketState(t::FloatType,S::Storage,Dynamics::MarketDynamics)
    X=Dynamics.Initialrate*ones(Dynamics.NF-1)
    X0=Dynamics.Initialrate
    OCM=OneCurrency(S,X0,X)
    return MarketState(S,OCM)
end

function Observations(t::FloatType,S::Storage,State::MarketState,MD::MarketDynamics)
    X0=State.USD.X0
    X=State.USD.X                                                                                                                                                                                                                                        
    ON=State.USD.X[end]+0.0005*randn()
    OCM=OneCurrencyObs(S,X0,X,ON)
    return Observations(S,OCM)
end

function ObservationsFromState!(t::FloatType,State::MarketState,out::Observations,MD::MarketDynamics)
    out.USD.X0=State.USD.X0
    out.USD.X=State.USD.X                                                                                                                                                                                                                                        
    out.USD.ON=State.USD.X[end]+0.0005*randn()
    return nothing
end

function UpdateState!(t0::FloatType,t1::FloatType,State::MarketState,NextState::MarketState,MD::MarketDynamics)
    kappa=[x for x in 1:(MD.NF-1)]
    kappa=MD.k1*((MD.b).^kappa)
    DT=(t1-t0)/360
    X0=State.USD.X0
    X=State.USD.X
    drift=[X0;X]
    X0+=[MD.s0*log(FloatType(1.0)+exp(MD.b0*only(X0)))*randn(FloatType)*sqrt(DT)]
    drift=-kappa.*diff(drift)*DT
    X+=drift+MD.s1*randn(MD.NF-1)*sqrt(DT)
    NextState.USD.X0=X0
    NextState.USD.X=X
    return nothing
end
function ExpectedObservations!(t1::FloatType,t2::FloatType,State::MarketState,out::Observations,MD::MarketDynamics)
    kappa=[x for x in 1:(MD.NF-1)]
    kappa=MD.k1*((MD.b).^kappa)
    DT=(t2-t1)/360
    out.USD.X0=State.USD.X0
    drift=[State.USD.X0;State.USD.X]
    drift=-kappa.*diff(drift)*DT
    out.USD.X=State.USD.X+drift                                                                                                                                                                                                                                    
    out.USD.ON=out.USD.X[end]
    return nothing
end


end

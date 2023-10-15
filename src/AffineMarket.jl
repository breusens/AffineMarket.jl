module AffineMarket
export AffineDynamics,AffineState
#include("VectorStoredArray.jl")
using Batcher
using MarketData
using Random
import MarketData: Observations
import MarketModels: MarketDynamics,MarketState,ObservationsFromState!,UpdateState!,ExpectedObservations! 
 
using FloatingNumberType


struct AffineDynamics<: MarketDynamics
    MarketFile::String
    NF::Int64
    Initialrate::SimType
    k1::SimType
    theta::SimType
    s1::SimType
    s0::SimType
    b0::SimType
    thetaQ::SimType
    b::SimType
    se2::SimType
end

struct OneCurrency <: MarketState
    X0::Batched
    X::Batched
end

struct AffineState <: MarketState
    USD::OneCurrency
end


function MarketState(t::Batched,CV::CA,Dynamics::AffineDynamics)
    X=CV(Dynamics.Initialrate*ones(Dynamics.NF-1))
    X0=CV(Dynamics.Initialrate)
    OCM=OneCurrency(X0,X)
    return AffineState(OCM)
end

function Observations(t::Batched,CV::CA,State::AffineState,MD::AffineDynamics)
    X0=State.USD.X0
    X=State.USD.X
    Noise=CV(SimType,(),randn)                                                                                                                                                                                                                                      
    ON=State.USD.X[end]+SimType(0.0005)*Noise
    OCM=OneCurrencyObs(X0,X,ON)
    return ObservedMarket(OCM)
end

function ObservationsFromState!(t::Batched,State::AffineState,out::ObservedMarket,MD::AffineDynamics,CV::CA)
    out.USD.X0.=State.USD.X0
    out.USD.X.=State.USD.X
    Noise=CV(SimType,(),randn)                                                                                                                                                                                                                                      
    out.USD.ON.=State.USD.X[end]+SimType(0.0005)*Noise
    return nothing
end

function UpdateState!(t0::Batched,t1::Batched,State::AffineState,NextState::AffineState,MD::AffineDynamics,CV::CA)
    kappa=[x for x in 1:(MD.NF-1)]
    kappa=MD.k1*((MD.b).^kappa)
    DT=(t1-t0)/SimType(360)
    X0=State.USD.X0
    X=State.USD.X
    drift0=[X0;X]
    Noise=CV(SimType,(),randn)
    X0+=MD.s0*log(SimType(1.0)+exp(MD.b0*X0))*Noise*sqrt(DT)
    drift=-kappa.*diff(drift0).*DT
    Noise2=CV(SimType,(MD.NF-1,),randn)
    X.+=drift.+MD.s1.*Noise2.*sqrt(DT)
    NextState.USD.X0.=X0
    NextState.USD.X.=X
    return nothing
end

function ExpectedObservations!(t1::Batched,t2::Batched,State::AffineState,out::ObservedMarket,MD::AffineDynamics,CV::CA)
    kappa=[x for x in 1:(MD.NF-1)]
    kappa=MD.k1*((MD.b).^kappa)
    DT=(t2-t1)/SimType(360)
    out.USD.X0.=State.USD.X0
    drift=[State.USD.X0;State.USD.X]
    d=-kappa.*diff(drift).*DT
    out.USD.X.=State.USD.X.+d                                                                                                                                                                                                                                  
    out.USD.ON.=out.USD.X[end]
    return nothing
end


end

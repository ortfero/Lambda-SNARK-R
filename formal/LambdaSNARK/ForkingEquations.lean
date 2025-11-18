/-
Helper abstraction for supplying concrete verifier equations required by the
forking extractor.  This module packages the data needed to produce a
`ProtocolForkingEquations` witness from circuit-specific algebraic facts.
-/

import LambdaSNARK.Core
import LambdaSNARK.ForkingInfrastructure

namespace LambdaSNARK

variable {F : Type} [Field F] [Fintype F] [DecidableEq F]
variable (VC : VectorCommitment F) (cs : R1CS F)

/--
A typeclass witnessing that the concrete constraint system `cs` provides
sufficient algebraic information to satisfy the verifier equations demanded by
`ForkingInfrastructure`.  Instances are expected to package the domain size,
primitive root, and quotient polynomial relations for every forked pair of
transcripts.
-/
class ForkingEquationWitness where
  /-- Square domain assumption required by the extractor.-/
  square : cs.nVars = cs.nCons
  /-- Produce the verifier-equation certificate for any valid fork.-/
  buildCore :
    (t1 t2 : Transcript F VC) →
    (h_fork : is_valid_fork VC t1 t2) →
    ForkingVerifierEquationsCore VC cs t1 t2 h_fork

namespace ForkingEquationWitness

variable {VC cs}

/-- Promote a circuit witness to the `ProtocolForkingEquations` structure.-/
noncomputable def protocol (inst : ForkingEquationWitness VC cs) :
    ProtocolForkingEquations VC cs :=
  { square := inst.square
    buildCore := inst.buildCore }

/-- Promote a circuit witness to a `ForkingEquationsProvider`.-/
noncomputable def provider (inst : ForkingEquationWitness VC cs) :
    ForkingEquationsProvider VC cs :=
  ForkingEquationsProvider.ofProtocol (proto := protocol inst)

/--
Retrieve the protocol witness using typeclass inference.
-/
noncomputable def protocolOf [inst : ForkingEquationWitness VC cs] :
    ProtocolForkingEquations VC cs :=
  protocol (VC := VC) (cs := cs) inst

/--
Retrieve the provider witness using typeclass inference.
-/
noncomputable def providerOf [inst : ForkingEquationWitness VC cs] :
    ForkingEquationsProvider VC cs :=
  provider (VC := VC) (cs := cs) inst

end ForkingEquationWitness

end LambdaSNARK

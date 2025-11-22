import LambdaSNARK.Core
import LambdaSNARK.Polynomial

#check LambdaSNARK.constraintNumeratorPoly
#check LambdaSNARK.SoundnessAssumptions

variable {F : Type} [Field F]
variable (VC : LambdaSNARK.VectorCommitment F) (cs : LambdaSNARK.R1CS F)

#check LambdaSNARK.SoundnessAssumptions F VC cs

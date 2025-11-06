import Lake
open Lake DSL

package «LambdaSNARK» where
  -- add package configuration options here

lean_lib «LambdaSNARK» where
  -- add library configuration options here

@[default_target]
lean_exe «lambdasnark» where
  root := `Main

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

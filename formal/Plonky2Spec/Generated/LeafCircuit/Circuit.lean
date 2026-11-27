/-
  AUTO-GENERATED — do not edit by hand.

  Part of `Plonky2Spec.Generated.LeafCircuit`, split out of the generated module
  so each Lean module elaborates in bounded memory.
  Regenerate with:

      cargo run -p qp-plonky2-constraint-exporter --bin export-constraints
-/
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.LinearCombination
import Plonky2Spec.WiringGadgets
import Plonky2Spec.WiringSponge
import Plonky2Spec.Generated.LeafCircuit.CopyData

namespace Plonky2Spec.Generated

open Plonky2Spec.Wiring
open Plonky2Spec.Poseidon2 (St)
open Plonky2Spec.Sponge (spongeHash pad10 addBlock rate absorbMsg_block8 absorbMsg_nil)

set_option maxRecDepth 200000

set_option maxHeartbeats 0

set_option linter.all false

set_option Elab.async false

variable {p : ℕ} [Fact p.Prime]

/-- The wormhole leaf circuit (`wormhole/circuit/src/circuit.rs`): the unspendable account, the zk-tree Merkle proof, the block header and the shared-target wiring with the nullifier and block-hash bindings, on the witness targets `secret`, `transfer_count`, `to_account`, `account_id`, `root_hash`, `depth`, `positions`, `is_not_dummy`, `siblings_0..15` and the header fields. -/
noncomputable def leafCircuit (p : ℕ) : Circuit p where
  rows := [
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 0
    ⟨.poseidon2, []⟩,  -- row 1
    ⟨.poseidon2, []⟩,  -- row 2
    ⟨.baseSum2 59, []⟩,  -- row 3
    ⟨.baseSum2 59, []⟩,  -- row 4
    ⟨.baseSum2 59, []⟩,  -- row 5
    ⟨.baseSum2 59, []⟩,  -- row 6
    ⟨.poseidon2, []⟩,  -- row 7
    ⟨.arithmetic 15, [1, 1]⟩,  -- row 8
    ⟨.poseidon2, []⟩,  -- row 9
    ⟨.baseSum2 59, []⟩,  -- row 10
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 11
    ⟨.arithmetic 15, [(-1), 1]⟩,  -- row 12
    ⟨.baseSum2 59, []⟩,  -- row 13
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 14
    ⟨.baseSum2 59, []⟩,  -- row 15
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 16
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 17
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 18
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 19
    ⟨.poseidon2, []⟩,  -- row 20
    ⟨.arithmetic 15, [1, 1]⟩,  -- row 21
    ⟨.poseidon2, []⟩,  -- row 22
    ⟨.poseidon2, []⟩,  -- row 23
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 24
    ⟨.baseSum2 59, []⟩,  -- row 25
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 26
    ⟨.baseSum2 59, []⟩,  -- row 27
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 28
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 29
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 30
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 31
    ⟨.poseidon2, []⟩,  -- row 32
    ⟨.arithmetic 15, [1, 1]⟩,  -- row 33
    ⟨.poseidon2, []⟩,  -- row 34
    ⟨.poseidon2, []⟩,  -- row 35
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 36
    ⟨.baseSum2 59, []⟩,  -- row 37
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 38
    ⟨.baseSum2 59, []⟩,  -- row 39
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 40
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 41
    ⟨.arithmetic 15, [(-1), 1]⟩,  -- row 42
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 43
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 44
    ⟨.poseidon2, []⟩,  -- row 45
    ⟨.arithmetic 15, [1, 1]⟩,  -- row 46
    ⟨.poseidon2, []⟩,  -- row 47
    ⟨.poseidon2, []⟩,  -- row 48
    ⟨.baseSum2 59, []⟩,  -- row 49
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 50
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 51
    ⟨.baseSum2 59, []⟩,  -- row 52
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 53
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 54
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 55
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 56
    ⟨.poseidon2, []⟩,  -- row 57
    ⟨.poseidon2, []⟩,  -- row 58
    ⟨.arithmetic 15, [1, 1]⟩,  -- row 59
    ⟨.poseidon2, []⟩,  -- row 60
    ⟨.baseSum2 59, []⟩,  -- row 61
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 62
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 63
    ⟨.baseSum2 59, []⟩,  -- row 64
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 65
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 66
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 67
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 68
    ⟨.poseidon2, []⟩,  -- row 69
    ⟨.poseidon2, []⟩,  -- row 70
    ⟨.poseidon2, []⟩,  -- row 71
    ⟨.baseSum2 59, []⟩,  -- row 72
    ⟨.arithmetic 15, [1, 1]⟩,  -- row 73
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 74
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 75
    ⟨.baseSum2 59, []⟩,  -- row 76
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 77
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 78
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 79
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 80
    ⟨.poseidon2, []⟩,  -- row 81
    ⟨.poseidon2, []⟩,  -- row 82
    ⟨.poseidon2, []⟩,  -- row 83
    ⟨.baseSum2 59, []⟩,  -- row 84
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 85
    ⟨.arithmetic 15, [1, 1]⟩,  -- row 86
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 87
    ⟨.baseSum2 59, []⟩,  -- row 88
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 89
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 90
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 91
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 92
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 93
    ⟨.poseidon2, []⟩,  -- row 94
    ⟨.poseidon2, []⟩,  -- row 95
    ⟨.poseidon2, []⟩,  -- row 96
    ⟨.baseSum2 59, []⟩,  -- row 97
    ⟨.arithmetic 15, [1, 1]⟩,  -- row 98
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 99
    ⟨.baseSum2 59, []⟩,  -- row 100
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 101
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 102
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 103
    ⟨.arithmetic 15, [(-1), 1]⟩,  -- row 104
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 105
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 106
    ⟨.poseidon2, []⟩,  -- row 107
    ⟨.poseidon2, []⟩,  -- row 108
    ⟨.poseidon2, []⟩,  -- row 109
    ⟨.baseSum2 59, []⟩,  -- row 110
    ⟨.arithmetic 15, [1, 1]⟩,  -- row 111
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 112
    ⟨.baseSum2 59, []⟩,  -- row 113
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 114
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 115
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 116
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 117
    ⟨.poseidon2, []⟩,  -- row 118
    ⟨.poseidon2, []⟩,  -- row 119
    ⟨.poseidon2, []⟩,  -- row 120
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 121
    ⟨.baseSum2 59, []⟩,  -- row 122
    ⟨.baseSum2 59, []⟩,  -- row 123
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 124
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 125
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 126
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 127
    ⟨.arithmetic 15, [1, 1]⟩,  -- row 128
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 129
    ⟨.poseidon2, []⟩,  -- row 130
    ⟨.poseidon2, []⟩,  -- row 131
    ⟨.poseidon2, []⟩,  -- row 132
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 133
    ⟨.baseSum2 59, []⟩,  -- row 134
    ⟨.baseSum2 59, []⟩,  -- row 135
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 136
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 137
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 138
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 139
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 140
    ⟨.poseidon2, []⟩,  -- row 141
    ⟨.arithmetic 15, [1, 1]⟩,  -- row 142
    ⟨.poseidon2, []⟩,  -- row 143
    ⟨.poseidon2, []⟩,  -- row 144
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 145
    ⟨.baseSum2 59, []⟩,  -- row 146
    ⟨.baseSum2 59, []⟩,  -- row 147
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 148
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 149
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 150
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 151
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 152
    ⟨.poseidon2, []⟩,  -- row 153
    ⟨.arithmetic 15, [1, 1]⟩,  -- row 154
    ⟨.poseidon2, []⟩,  -- row 155
    ⟨.poseidon2, []⟩,  -- row 156
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 157
    ⟨.baseSum2 59, []⟩,  -- row 158
    ⟨.baseSum2 59, []⟩,  -- row 159
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 160
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 161
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 162
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 163
    ⟨.arithmetic 15, [(-1), 1]⟩,  -- row 164
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 165
    ⟨.poseidon2, []⟩,  -- row 166
    ⟨.arithmetic 15, [1, 1]⟩,  -- row 167
    ⟨.poseidon2, []⟩,  -- row 168
    ⟨.poseidon2, []⟩,  -- row 169
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 170
    ⟨.baseSum2 59, []⟩,  -- row 171
    ⟨.baseSum2 59, []⟩,  -- row 172
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 173
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 174
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 175
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 176
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 177
    ⟨.poseidon2, []⟩,  -- row 178
    ⟨.arithmetic 15, [1, 1]⟩,  -- row 179
    ⟨.poseidon2, []⟩,  -- row 180
    ⟨.poseidon2, []⟩,  -- row 181
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 182
    ⟨.baseSum2 59, []⟩,  -- row 183
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 184
    ⟨.baseSum2 59, []⟩,  -- row 185
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 186
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 187
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 188
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 189
    ⟨.poseidon2, []⟩,  -- row 190
    ⟨.arithmetic 15, [1, 1]⟩,  -- row 191
    ⟨.poseidon2, []⟩,  -- row 192
    ⟨.poseidon2, []⟩,  -- row 193
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 194
    ⟨.baseSum2 59, []⟩,  -- row 195
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 196
    ⟨.baseSum2 59, []⟩,  -- row 197
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 198
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 199
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 200
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 201
    ⟨.poseidon2, []⟩,  -- row 202
    ⟨.arithmetic 15, [1, 1]⟩,  -- row 203
    ⟨.poseidon2, []⟩,  -- row 204
    ⟨.poseidon2, []⟩,  -- row 205
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 206
    ⟨.baseSum2 59, []⟩,  -- row 207
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 208
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 209
    ⟨.poseidon2, []⟩,  -- row 210
    ⟨.poseidon2, []⟩,  -- row 211
    ⟨.poseidon2, []⟩,  -- row 212
    ⟨.poseidon2, []⟩,  -- row 213
    ⟨.poseidon2, []⟩,  -- row 214
    ⟨.arithmetic 15, [1, 1]⟩,  -- row 215
    ⟨.poseidon2, []⟩,  -- row 216
    ⟨.poseidon2, []⟩,  -- row 217
    ⟨.arithmetic 15, [1, 1]⟩,  -- row 218
    ⟨.poseidon2, []⟩,  -- row 219
    ⟨.poseidon2, []⟩,  -- row 220
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 221
    ⟨.arithmetic 15, [1, (-1)]⟩  -- row 222
  ]
  copies := ((((((((leafCircuit.copies0 ++ leafCircuit.copies1) ++ (leafCircuit.copies2 ++ leafCircuit.copies3)) ++ ((leafCircuit.copies4 ++ leafCircuit.copies5) ++ (leafCircuit.copies6 ++ leafCircuit.copies7))) ++ (((leafCircuit.copies8 ++ leafCircuit.copies9) ++ (leafCircuit.copies10 ++ leafCircuit.copies11)) ++ ((leafCircuit.copies12 ++ leafCircuit.copies13) ++ (leafCircuit.copies14 ++ leafCircuit.copies15)))) ++ ((((leafCircuit.copies16 ++ leafCircuit.copies17) ++ (leafCircuit.copies18 ++ leafCircuit.copies19)) ++ ((leafCircuit.copies20 ++ leafCircuit.copies21) ++ (leafCircuit.copies22 ++ leafCircuit.copies23))) ++ (((leafCircuit.copies24 ++ leafCircuit.copies25) ++ (leafCircuit.copies26 ++ leafCircuit.copies27)) ++ ((leafCircuit.copies28 ++ leafCircuit.copies29) ++ (leafCircuit.copies30 ++ leafCircuit.copies31))))) ++ (((((leafCircuit.copies32 ++ leafCircuit.copies33) ++ (leafCircuit.copies34 ++ leafCircuit.copies35)) ++ ((leafCircuit.copies36 ++ leafCircuit.copies37) ++ (leafCircuit.copies38 ++ leafCircuit.copies39))) ++ (((leafCircuit.copies40 ++ leafCircuit.copies41) ++ (leafCircuit.copies42 ++ leafCircuit.copies43)) ++ ((leafCircuit.copies44 ++ leafCircuit.copies45) ++ (leafCircuit.copies46 ++ leafCircuit.copies47)))) ++ ((((leafCircuit.copies48 ++ leafCircuit.copies49) ++ (leafCircuit.copies50 ++ leafCircuit.copies51)) ++ ((leafCircuit.copies52 ++ leafCircuit.copies53) ++ (leafCircuit.copies54 ++ leafCircuit.copies55))) ++ (((leafCircuit.copies56 ++ leafCircuit.copies57) ++ (leafCircuit.copies58 ++ leafCircuit.copies59)) ++ ((leafCircuit.copies60 ++ leafCircuit.copies61) ++ (leafCircuit.copies62 ++ (leafCircuit.copies63 ++ leafCircuit.copies64))))))) ++ ((((((leafCircuit.copies65 ++ leafCircuit.copies66) ++ (leafCircuit.copies67 ++ leafCircuit.copies68)) ++ ((leafCircuit.copies69 ++ leafCircuit.copies70) ++ (leafCircuit.copies71 ++ leafCircuit.copies72))) ++ (((leafCircuit.copies73 ++ leafCircuit.copies74) ++ (leafCircuit.copies75 ++ leafCircuit.copies76)) ++ ((leafCircuit.copies77 ++ leafCircuit.copies78) ++ (leafCircuit.copies79 ++ leafCircuit.copies80)))) ++ ((((leafCircuit.copies81 ++ leafCircuit.copies82) ++ (leafCircuit.copies83 ++ leafCircuit.copies84)) ++ ((leafCircuit.copies85 ++ leafCircuit.copies86) ++ (leafCircuit.copies87 ++ leafCircuit.copies88))) ++ (((leafCircuit.copies89 ++ leafCircuit.copies90) ++ (leafCircuit.copies91 ++ leafCircuit.copies92)) ++ ((leafCircuit.copies93 ++ leafCircuit.copies94) ++ (leafCircuit.copies95 ++ (leafCircuit.copies96 ++ leafCircuit.copies97)))))) ++ (((((leafCircuit.copies98 ++ leafCircuit.copies99) ++ (leafCircuit.copies100 ++ leafCircuit.copies101)) ++ ((leafCircuit.copies102 ++ leafCircuit.copies103) ++ (leafCircuit.copies104 ++ leafCircuit.copies105))) ++ (((leafCircuit.copies106 ++ leafCircuit.copies107) ++ (leafCircuit.copies108 ++ leafCircuit.copies109)) ++ ((leafCircuit.copies110 ++ leafCircuit.copies111) ++ (leafCircuit.copies112 ++ leafCircuit.copies113)))) ++ ((((leafCircuit.copies114 ++ leafCircuit.copies115) ++ (leafCircuit.copies116 ++ leafCircuit.copies117)) ++ ((leafCircuit.copies118 ++ leafCircuit.copies119) ++ (leafCircuit.copies120 ++ leafCircuit.copies121))) ++ (((leafCircuit.copies122 ++ leafCircuit.copies123) ++ (leafCircuit.copies124 ++ leafCircuit.copies125)) ++ ((leafCircuit.copies126 ++ leafCircuit.copies127) ++ (leafCircuit.copies128 ++ (leafCircuit.copies129 ++ leafCircuit.copies130)))))))) ++ (((((((leafCircuit.copies131 ++ leafCircuit.copies132) ++ (leafCircuit.copies133 ++ leafCircuit.copies134)) ++ ((leafCircuit.copies135 ++ leafCircuit.copies136) ++ (leafCircuit.copies137 ++ leafCircuit.copies138))) ++ (((leafCircuit.copies139 ++ leafCircuit.copies140) ++ (leafCircuit.copies141 ++ leafCircuit.copies142)) ++ ((leafCircuit.copies143 ++ leafCircuit.copies144) ++ (leafCircuit.copies145 ++ leafCircuit.copies146)))) ++ ((((leafCircuit.copies147 ++ leafCircuit.copies148) ++ (leafCircuit.copies149 ++ leafCircuit.copies150)) ++ ((leafCircuit.copies151 ++ leafCircuit.copies152) ++ (leafCircuit.copies153 ++ leafCircuit.copies154))) ++ (((leafCircuit.copies155 ++ leafCircuit.copies156) ++ (leafCircuit.copies157 ++ leafCircuit.copies158)) ++ ((leafCircuit.copies159 ++ leafCircuit.copies160) ++ (leafCircuit.copies161 ++ (leafCircuit.copies162 ++ leafCircuit.copies163)))))) ++ (((((leafCircuit.copies164 ++ leafCircuit.copies165) ++ (leafCircuit.copies166 ++ leafCircuit.copies167)) ++ ((leafCircuit.copies168 ++ leafCircuit.copies169) ++ (leafCircuit.copies170 ++ leafCircuit.copies171))) ++ (((leafCircuit.copies172 ++ leafCircuit.copies173) ++ (leafCircuit.copies174 ++ leafCircuit.copies175)) ++ ((leafCircuit.copies176 ++ leafCircuit.copies177) ++ (leafCircuit.copies178 ++ leafCircuit.copies179)))) ++ ((((leafCircuit.copies180 ++ leafCircuit.copies181) ++ (leafCircuit.copies182 ++ leafCircuit.copies183)) ++ ((leafCircuit.copies184 ++ leafCircuit.copies185) ++ (leafCircuit.copies186 ++ leafCircuit.copies187))) ++ (((leafCircuit.copies188 ++ leafCircuit.copies189) ++ (leafCircuit.copies190 ++ leafCircuit.copies191)) ++ ((leafCircuit.copies192 ++ leafCircuit.copies193) ++ (leafCircuit.copies194 ++ (leafCircuit.copies195 ++ leafCircuit.copies196))))))) ++ ((((((leafCircuit.copies197 ++ leafCircuit.copies198) ++ (leafCircuit.copies199 ++ leafCircuit.copies200)) ++ ((leafCircuit.copies201 ++ leafCircuit.copies202) ++ (leafCircuit.copies203 ++ leafCircuit.copies204))) ++ (((leafCircuit.copies205 ++ leafCircuit.copies206) ++ (leafCircuit.copies207 ++ leafCircuit.copies208)) ++ ((leafCircuit.copies209 ++ leafCircuit.copies210) ++ (leafCircuit.copies211 ++ leafCircuit.copies212)))) ++ ((((leafCircuit.copies213 ++ leafCircuit.copies214) ++ (leafCircuit.copies215 ++ leafCircuit.copies216)) ++ ((leafCircuit.copies217 ++ leafCircuit.copies218) ++ (leafCircuit.copies219 ++ leafCircuit.copies220))) ++ (((leafCircuit.copies221 ++ leafCircuit.copies222) ++ (leafCircuit.copies223 ++ leafCircuit.copies224)) ++ ((leafCircuit.copies225 ++ leafCircuit.copies226) ++ (leafCircuit.copies227 ++ (leafCircuit.copies228 ++ leafCircuit.copies229)))))) ++ (((((leafCircuit.copies230 ++ leafCircuit.copies231) ++ (leafCircuit.copies232 ++ leafCircuit.copies233)) ++ ((leafCircuit.copies234 ++ leafCircuit.copies235) ++ (leafCircuit.copies236 ++ leafCircuit.copies237))) ++ (((leafCircuit.copies238 ++ leafCircuit.copies239) ++ (leafCircuit.copies240 ++ leafCircuit.copies241)) ++ ((leafCircuit.copies242 ++ leafCircuit.copies243) ++ (leafCircuit.copies244 ++ leafCircuit.copies245)))) ++ ((((leafCircuit.copies246 ++ leafCircuit.copies247) ++ (leafCircuit.copies248 ++ leafCircuit.copies249)) ++ ((leafCircuit.copies250 ++ leafCircuit.copies251) ++ (leafCircuit.copies252 ++ leafCircuit.copies253))) ++ (((leafCircuit.copies254 ++ leafCircuit.copies255) ++ (leafCircuit.copies256 ++ leafCircuit.copies257)) ++ ((leafCircuit.copies258 ++ leafCircuit.copies259) ++ (leafCircuit.copies260 ++ (leafCircuit.copies261 ++ leafCircuit.copies262)))))))))
  constants := [
    (.virt 14, 0),
    (.virt 290, 1836216183),
    (.virt 291, 1701605224),
    (.virt 292, 1),
    (.virt 293, 576460752303423488 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 294, 2),
    (.virt 295, 3),
    (.virt 432, 1819635326),
    (.virt 433, 2120640876)
  ]
  publicInputs := [
    .virt 0,
    .virt 223,
    .virt 224,
    .virt 225,
    .virt 226,
    .virt 241,
    .virt 242,
    .virt 243,
    .virt 244,
    .virt 249,
    .virt 7
  ]

end Plonky2Spec.Generated

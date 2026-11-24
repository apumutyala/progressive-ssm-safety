# Source provenance

Original source commit `e137aff851daf5cb0edecd5c9e5835f903c3d4dc`, tree `067fe855962b3c393591a4200bbd89e092683504`. This snapshot contains code shared by the earlier corpus growth, evidence-response, four-arm grounding and causal diagnosis studies.

## Exact files

| File | Original and release SHA-256 |
| --- | --- |
| `authority_cases.py` | `5d7321441c280fe5e550417261c752a2e58263a79bc10fd5bb275055ff614420` |
| `causal_trace_core.py` | `18c401a74394cd9a26e7bd50412deb2a00b3ebb53e07d5a1c9f283a7dc231dc5` |
| `comparator.py` | `36e8f6bbfc96824439989485f988a8a81c7b3205157a79b4eeefe983664ee1b8` |
| `control_probe.py` | `7832bf8c238c72ff057bf589cb9df2a5e78940c35203295a7e51ebdc7666d674` |
| `corpus_io.py` | `94686a02382301221c99952a03800041ae21abafa7a276ecdd9afc15fc2ba429` |
| `corpus_model.py` | `08e436529335138395580e3b28db011257ac48969266261c5f0c49725fe14c93` |
| `evidence_control.py` | `66deb1fa830d43d6dffb5ab855f75d8968bc32ea02dd61a2e1598fd70520a902` |
| `evidence_pairs.py` | `b39e7d184f68304fbdf4e26b44261749bceed37cfc4dc8a11a83ee73f37232a6` |
| `evidence_recipe.py` | `5c39a31da6f553ad734fa84d502b0bbc27275930ae893812937f1690228038ad` |
| `grammar_prefix_bound.py` | `fce181d96cbf6b7f04c1a931573fde55c9166cc4fc6e01652d0986b16d37b029` |
| `interval_model.py` | `421f17b442105dea44fa44192f3c87be518a5f85d211b2edb8210c9296cc536a` |
| `model.py` | `61ca237e6296ddc79ce02fb457ae62087f9724517ffc1a8d4c8492df09229abf` |
| `pilot_evaluation.py` | `9ffae2206728df2a71a31862e5fe58c7aa1809c1e5e057dba1357f8c07c660d8` |
| `response_step.py` | `a1bf254f35cab4a5b4bcd28fade7ab08b543ce2017af2d89d67bcfadd629bd46` |
| `test_causal_input_audit.py` | `d8b1ead324e42bfca40f91cdfad9631e13b378c5fc5d49417cbb84ac640e0a32` |
| `test_causal_trace_core.py` | `facc211871ad3d6318e972c93b746430525ba852fc32f4d635e350f7ac169173` |
| `test_comparator_contract.py` | `3d23f6a7d478c08dedb5b0692767bc76a8e68e65caf61773b7ff0d4a0c131870` |
| `test_evidence_control.py` | `96ccc40d153ea70484ab774336b86806e6ebc73a8f662e48b97b3e23d7e98897` |
| `test_evidence_recipe.py` | `c6ac9caaba17e8fad58effb4434e23bc59785b69dc379bc3d3fc0fdac12fde5f` |
| `test_grammar_prefix_bound.py` | `405093c10d571e1e8e74e4969a7725d5e89e8a15c01c8a367667ac01f11a8db9` |
| `test_pilot_evaluation.py` | `5d9feb3ecac5440dbe6a7aa6a23b7bb1cc98633138f3b4b582ef067dc169a489` |
| `test_response_step.py` | `b1d1275896bb259a7cf92fa88d7664942046c4a7db3d5e161285b13949d43ea6` |

## Source excerpts

The following release modules contain original declarations with portable imports. Each declaration preserves the original source lines, including any decorators. The full archived jobs and private-evaluation interfaces remain local.

- `growth_transfer.py` (`ee47bb655027d574ed17e9c9e7d63d9ae4c928911f9365ff11f74a6366172346`) from `growth_train.py` (`e3d7fb5a3ffae3da9632ddac0a9b071d1a1433aaf42c898ddd1312619418dede`), declarations `bits_equal`, `tensor_changes`, `decision_probe`, `insert`.
- `pretrain.py` (`20039ec2aa81bd94fe94c0dd6a272ef5d4835fae87f2dae39326c4567183ce07`) from `pretrain.py` (`8a11f9ba2354fd01a28bd5d19d67e665a1902b0d812ec4fe0ea4a154cf89502c`), declarations `exact_equal`, `finite`, `audit_ordinary_output`, `pack`.
- `response_eval.py` (`0938933510d3a57b97a6cf5add2923e46e7b4e8ae7082a658cfd03342975b501`) from `response_eval.py` (`cd58353becd6da49a6d8cd4c4bac69b1a591bc843875ea23766c2db0d5b2a4c0`), declarations `LABELS`, `MAX_RAW_TOKENS`, `_normal`, `_validate_expected`, `parse_response`, `score_case`.
- `causal_input_audit.py` (`28767457889578b6fc530b4918e6db05ca4141a7ab561f84fea0a4a27135135a`) from `causal_input_audit.py` (`386cf1ae0c94b4d75bb544ac12205d0e0cbaf73c4748b7f9bcc8c3f27b4051dc`), declarations `need`, `canonical`, `sha`, `check_shift`, `check_matching`, `source_contract`.

The README, provenance manifest and Git attributes are release-specific.

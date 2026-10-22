# Restart: conditional residual without anti-guidance

Authorization: restart training after removing the anti-guidance target; retain UND architecture, do not implement GEN velocity output. Reuse previous 64NPU/5000step/batch256/512px/lr1e-4/warmup100/seed17 and the accepted photo corpus. Fresh native initialization, new output directory; preserve previous weights/images/data.

Let g=v_guided-v_conditional and H_g(z)=z-min(<z,g>/||g||²,0)g (identity when g=0).

New target is y=H_g(epsilon-x0-v_conditional), NOT epsilon-x0-v_guided. Train raw UND output with MSE(raw_delta,y). Raw prediction is not projected before loss, so negative raw components get corrective gradients instead of a flat loss region. Project the actual deployed correction with the SAME H_g, including the global baseline. Proposal input stays x_t-t*v_guided, backbone/features/time range remain unchanged. This deliberately restricts the target to guidance-preserving residuals; it is no longer full guided-flow regression.

Metrics: loss=raw prediction vs projected conditional target; zero_correction_mse is its zero-output baseline. Retain base_mse and validation residual_mse as physical guided velocity error, and log conditional-flow errors and raw/applied guidance projections separately. Never interpret loss/base_mse as a matched percentage improvement. Track target projection magnitude and applied anti-guidance coefficient.

Acceptance: projection geometry/zero guidance, target formula and gradients, old checkpoint sampling semantics, exact new-mode resume/global baseline state, zero-init native parity, full NPU pilot and actual pictures; then launch 4nodes*16NPU and verify64ranks plus checkpoint10. Fixed32-prompt three-arm images at100/1000/3000/5000. Goal is a running accepted experiment, not a preclaimed semantic quality improvement.

# Memo to Ritu Deshpande
**Subject:** Kestrel service-request routing bot renewal

## Decision
Do not renew the Rs 3.2 lakh/year vendor routing bot. Use the replacement router and keep a human override for low-confidence cases during cutover.

## The number
On the most relevant pre-test time holdout (May-Jun 2026), the replacement matched the historical routing label **95.48%** of the time, versus the 90% requirement. The miss rate was **4.52%**. A separate random holdout was 95.57%, so the result is stable rather than a single lucky split.

## The rupees
The vendor licence costs **Rs 3.2 lakh/year**. Across the 18-month history, **2,696 of 10,822 requests (24.91%)** had at least one transfer, producing **3,902 transfer events**. At the policy's costs, that is **Rs 18.91 lakh of routing-related cost exposure over 18 months**, or about **Rs 12.61 lakh annualized**, before the licence. Combined direct cost exposure is therefore about **Rs 15.81 lakh/year**.

Those routing-cost figures are historical exposure, not guaranteed future savings. The licence saving is the cleanest direct saving: **Rs 3.2 lakh/year** if the vendor bot is not renewed.

## What to do next week
**Monday:** deploy the replacement in parallel with the current flow and watch a live sample.

**Tuesday-Wednesday:** review the low-confidence and mixed-intent requests; make sure the manual override is visible to agents.

**Thursday:** switch the vendor routing bot off if the monitored sample remains above 90% match.

**Friday:** review transfer rate, exception rate and agent feedback; keep the simple local model unless those numbers deteriorate.

The main weakness is not infrastructure. It is ambiguous historical labelling: some repeated request texts have conflicting team labels. That should be monitored rather than hidden.

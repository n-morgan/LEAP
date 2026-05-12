# GPT quotes, RLM summarizes — Miami-Dade examples

---

## 1. Retrofit homes

**Source:** Retrofit 167,500 homes to reduce energy costs 28%, prioritizing Low- to Moderate-Income (LMI) homes by 2030

**GT:** Retrofit 167,500 homes to reduce energy costs by 28%, prioritizing low- to moderate-income homes, by 2030

- **GPT (sim=0.99, grade=+1):** Retrofit 167,500 homes by 2030 to reduce energy costs by 28%, prioritizing low- to moderate-income homes.
- **RLM (sim=0.63, grade=0):** Benchmark, retune, and retrofit existing buildings to reduce building energy waste.

RLM collapsed three separate bullet points (benchmark, retune, retrofit) into one generic statement and lost the 167,500 target, 28% cost reduction, and 2030 deadline.

---

## 2. Utility-scale renewable energy

**Source:** MAXIMIZE PARTICIPATION IN UTILITY-SCALE RENEWABLE ENERGY PROGRAMS

**GT:** Maximize participation in utility-scale renewable energy programs.

- **GPT (sim=1.0, grade=+1):** Maximize participation in utility-scale renewable energy programs.
- **RLM (sim=0.61, grade=0):** Expand on-site and off-site renewable energy generation.

RLM paraphrased the broader renewable energy section and changed the mechanism from participating in utility-scale programs to expanding generation — different policy, different actor.

---

## 3. Waste to energy

**Source:** 50% of non-recycled garbage converted to energy by 2030

**GT:** Convert 50% of non-recycled garbage to energy by 2030.

- **GPT (sim=1.0, grade=+1):** Convert 50% of non-recycled garbage to energy by 2030.
- **RLM (sim=0.62, grade=0):** Convert waste to energy to reduce methane emissions, generate electricity, and reduce landfill disposal.

RLM picked up the rationale bullets below the target (methane, electricity, landfill) and merged them into the policy statement, dropping the 50% target and 2030 deadline entirely.

---

## 4. Double natural habitat

**Source:** Double the total non-wetland acreage of natural habitat in preservation

**GT:** Double the total non-wetland acreage of natural habitat in preservation.

- **GPT (sim=1.0, grade=+1):** Double the total non-wetland acreage of natural habitat in preservation.
- **RLM (sim=0.67, grade=0):** Reduce pollutant loads to surface waters to restore seagrasses to historic levels and double non-wetland natural habitat acreage in preservation.

RLM fused two adjacent bullets into one sentence. The seagrass restoration commitment is a separate policy. The grader correctly rejected the fused version.

---

## 5. Water consumption

**Source:** Reduce water consumption per person 30% by 2030

**GT:** Reduce water consumption per person 30% by 2030.

- **GPT (sim=0.99, grade=+1):** Reduce water consumption per person by 30% by 2030.
- **RLM (sim=0.57, grade=0):** Reduce waste generation and water use to reduce emissions, infrastructure costs, and environmental impacts.

RLM merged the water bullet with the landfill waste bullet directly above it. The 30% target and 2030 deadline are gone, and waste generation is now part of the same sentence as water use.

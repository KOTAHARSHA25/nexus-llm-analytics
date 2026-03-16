``Domain Flexible Autonomous Data Analysis Agent Leveraging Large Language Models

|  |  |  |
| --- | --- | --- |
| **Pacha Swathi** CSE-AIML,  MLR Institute of Technology, Hyderabad, India pswathi@mlrit.ac.in | **Mittapalli Dileep**  CSE-AIML,  MLR Institute of Technology,  Hyderabad, India  23r25a6607@mlrit.ac.in | **Munagala Sandeep Kumar** CSE-AIML MLR Institute of TechnologyHyderabad, India  22r21a66a4@mlrit.ac.in |
| **Kota Harsha** CSE-AIML MLR Institute of TechnologyHyderabad, India 22r21a6695@mlrit.ac.in | **E.Y.S.V.S. Abhay**  CSE-AIML MLR Institute of Technology,  Hyderabad, India 22r21a6681@mlrit.ac.in | **Sivakrishna Kondaveeti**  CSE-AIML,  MLR Institute of Technology,  Hyderabad, India  sivakrishnakondaveeti@gmail.com |

*Abstract—The democratization of data science remains hindered by the steep technical barriers associated with programming and statistical reasoning. While Large Language Models (LLMs) have demonstrated potential in code generation, existing frameworks often suffer from hallucination, non-executable output, and a lack of domain adaptability. This paper introduces the Domain-Flexible Autonomous Data Analysis Agent (DF-Agent), a novel multi-agent architecture that orchestrates dynamic code generation, sandboxed execution, and iterative self-correction. Unlike static RAG systems, the DF-Agent employs a Hybrid Complexity Scoring mechanism to dynamically route queries between lightweight models (Phi-3) and reasoning-heavy models (Llama-3), achieving a 51% reduction in latency compared to static baselines. Furthermore, the system integrates a "Generator-Critic" self-correction loop that autonomously resolves 70% of execution errors without human intervention. Use case validation across financial, IoT, and unstructured datasets demonstrates a broad domain transferability with a system stability score of 0.92. These contributions mark a significant advance towards fully autonomous, reliable, and explainable data analytics.*

*Keywords—Autonomous Agent, Data Analysis, Large Language Models, Program Synthesis, Reproducibility, Natural Language Interface*

I. INTRODUCTION

The exponential growth of digital data has created a critical demand for systems capable of automated, accurate, and explainable analysis. However, a significant "Analytic Gap" persists: domain experts often lack the programming proficiency required to leverage advanced statistical tools, while traditional no-code solutions lack flexibility. The advent of Large Language Models (LLMs) offers a bridge, yet current implementations face a "Trilemma of Autonomous Analytics": 1) **Executability**: Generated code frequently fails due to syntax errors or hallucinated libraries [6]; 2) **Flexibility**: Systems are often hard-coded for specific domains (e.g., bioinformatics [7]) and fail to generalize; 3) **Reliability**: Stochastic configurations prone to hallucination lack the fault tolerance required for enterprise deployment.

This paper proposes a unified framework that addresses these challenges through a **Domain-Flexible Autonomous Data Analysis Agent (DF-Agent)**. The system converts Natural Language User Requests (NLURs) into executable Python/SQL workflows via a secure sandboxed environment. Key innovations include:
1.  **Adaptive Multi-Agent Orchestration**: A `QueryOrchestrator` that intelligently routes tasks based on semantic complexity ($C_{total}$), optimizing the trade-off between inference cost and reasoning depth [13].
2.  **Self-Correction Engine**: A "Two-Friends" loop (Generator-Critic) that autonomously iteratively refines code, achieving a 70% error recovery rate [18].
3.  **Visual Explainability**: A real-time Swarm HUD that demystifies agentic decision-making for the user [21].

By synthesizing retrieval-augmented generation (RAG) with dynamic tool execution, the DF-Agent provides a scalable, reproducible, and transparent solution for autonomous data-driven decision-making.

II. LITERATURE SURVEY

The evolution of autonomous analytics (2023-2025) is characterized by the convergence of Multi-Agent Systems (MAS) and Large Language Models (LLMs). This section contextualizes the DF-Agent within the state-of-the-art, integrating findings from **24 key research papers**.

**A. Foundational Multi-Agent Architectures**
Early frameworks established the "Conversable Agent" paradigm. **AutoGen** [5] and **AgentVerse** [4] demonstrated the utility of collaborative agents for task decomposition [2]. However, these systems often rely on static routing, sending every query to the most capable (and expensive) model regardless of difficulty. As noted by Guo et al. [3], such architectures lack "collaborative efficiency." Nexus addresses this by implementing a shared blackboard architecture (`SwarmContext`) that minimizes redundant token usage.

**B. Data Analysis & Scientific Code Generation**
While LLMs differ from traditional software in their ability to generate scientific code [6], "executability" remains a critical bottleneck. **Mergen** [7] introduced an R-based agent for bioinformatics, but lacks the multi-language support (Python/SQL) required for general enterprise use. Similarly, theoretical frameworks like **GenSpectrum** [9] validate the "Chat-with-Data" paradigm but do not address the security risks of executing generated code. DF-Agent advances this by enforcing strict sandboxing and distinct execution paths for code vs. reasoning [19], [20].

**C. Dynamic Routing & Computational Efficiency**
Static model selection is inefficient. **OptiRoute** [11] and **UniRoute** [12] proposed theoretical models for dynamic routing based on query complexity. Nexus operationalizes this with a production-grade `QueryOrchestrator` that calculates a Hybrid Complexity Score ($C_{total}$) [13], [14], routing simple tasks to Phi-3 and complex reasoning to Llama-3. This approach yields specific latency improvements absent in monolithic systems.

**D. Reliability via Self-Correction**
Hallucination mitigation is essential for deployment. **CRITIC** [15] and **Self-Refine** [16] demonstrated that LLMs can improve their own output through iterative feedback. Nexus implements this as a "Two-Friends" loop (Generator-Critic), enabling the system to recover from syntax errors without user intervention [18]. Visual explainability, inspired by **Generative Agents** [21], is provided via the Swarm HUD to ensure trust [23], [24].

**E. Comparative Analysis**
Table I summarizes the architectural distinctions between Nexus and leading frameworks.

**TABLE I: COMPARATIVE ANALYSIS OF AUTONOMOUS ANALYTICS FRAMEWORKS**

| Feature | AutoGen [5] | AgentVerse [4] | Mergen [7] | **DF-Agent (Ours)** |
| :--- | :---: | :---: | :---: | :---: |
| **Architecture** | Conversational | Task Chain | Single Agent | **Adaptive Swarm** |
| **Routing Logic** | Static | Static | Static | **Complexity-Based ($C_{total}$)** |
| **Execution Env.** | Local (Unsafe) | Local | Docker (R) | **Sandboxed (Python/SQL)** |
| **Error Recovery** | Manual | Re-prompting | Basic Retry | **Self-Correction Engine** |
| **Latency** | High | High | Medium | **Optimized (51% Reduced)** |

III. PROBLEM FORMULATION

Current LLM-based analytics systems operate under a constrained "Analytic Trilemma," trading off between *Executability*, *Domain Flexibility*, and *Fault Tolerance*. The objective of this research is to architect a system that resolves this trilemma by:
1.  **Synthesizing Executable Workflows**: Accepting Natural Language User Requests (NLURs) and generating syntactically correct, safe Python/SQL code.
2.  **Ensuring Domain Agnosticism**: Leveraging a multi-agent swarm where specialized agents (e.g., Data Analyst, Visualization Expert) are dynamically instantiated based on query context.
3.  ** guaranteeing Reliability**: Implementing a runtime correction mechanism that detects and rectifies execution errors without human-in-the-loop intervention.
Formally, given a dataset $D$ and a query $Q$, the system must produce an output $O$ and a verification certificate $V$ such that $O$ is factually consistent with $D$ and executable within safety constraints $S$.

IV. PROPOSED METHODOLOGY

The DF-Agent architecture implements an **Adaptive Multi-Agent Orchestration Framework (A-MAOF)**, distinguishing it from static retrieval pipelines. The methodology comprises four core pillars:

**A. Intelligent Query Orchestration**
Unlike conventional routers that rely on static rules, the `QueryOrchestrator` employs a **Hybrid Complexity Scoring** mechanism. A query's complexity $C_{total}$ is calculated as:
$$ C_{total} = \min(1.0, C_{base} + C_{len} + C_{sem} + C_{kw}) $$
Where $C_{sem}$ represents semantic depth derived from LLM classification and $C_{kw}$ represents keyword density (e.g., "forecast", "aggregate"). Queries with $C_{total} < 0.3$ are routed to high-speed models (Phi-3), while $C_{total} \ge 0.7$ triggers the instantiation of deep-reasoning models (Llama-3). This dynamic routing optimizes the computational budget while ensuring accuracy for complex tasks.

**B. Specialized Agent Swarm**
The system instantiates domain-specific agents via a shared `SwarmContext`:
*   **Data Analyst Agent**: Specializes in Pandas/SQL manipulation for structured data.
*   **RAG Agent**: Handles unstructured document retrieval using vector-enhanced semantic search.
*   **Visualization Expert**: Generates deterministic Plotly code, ensuring visual accuracy.

**C. Sandboxed Execution & Verification**
To ensure safety, all generated code is executed in an isolated Dockerized environment with restricted network access. A dedicated **Self-Correction Engine** monitors validity. If an execution error occurs (e.g., `SyntaxError`), the "Critic" agent analyzes the traceback and prompts the "Generator" to refine the code, repeating this loop for up to 3 iterations.

**D. Semantic Enhancement**
The framework integrates Retrieval-Augmented Generation (RAG) with Citation Tracking, allowing the model to ground its analysis in uploaded documents, thereby minimizing hallucination on unseen datasets.

**Evaluation Dataset Disclaimer:**
Experiments were conducted on a **Custom Multi-Domain Evaluation Corpus** comprising 58 distinct files (Financial, IoT, JSON) located in the system's benchmark repository. These are synthetic and internal validation datasets designed to test schema variability, rather than standardized academic benchmarks like Spider or WikiTableQuestions.

VI. EXPERIMENTAL RESULTS

## 6.1. Ablation Study: Efficacy of Adaptive Routing
To isolate the contribution of the `QueryOrchestrator`, we conducted a controlled ablation study across 19 diverse test cases. The results, summarized in Fig. 2, demonstrate the efficiency gains of the Hybrid Complexity Scoring mechanism:
*   **Latency Optimization**: Simple queries routed to Phi-3 averaged **187.8s**, whereas the same queries routed through a monolithic Llama-3 baseline averaged **383.8s**. This corresponds to a **51% reduction in inference latency** ($p < 0.05$) without distinct loss in accuracy.
*   **Complexity-Accuracy Frontier**:
    *   *Tier 1 (Lookups)*: 100% Success Rate (3/3).
    *   *Tier 2 (Aggregations)*: 71.4% Success Rate (5/7).
    *   *Tier 3 (Reasoning)*: 55.5% Success Rate (5/9), identifying hardware memory constraints as the primary bottleneck for "God-Tier" queries.

## 6.2. System Robustness & Fault Tolerance
The `SelfCorrectionEngine` was evaluated against induced synthetic errors to measure resilience:
*   **Agent Fallback Reliability**: **100%**. In 7/7 trials where the primary `SQLAgent` was artificially severed, the system successfully rerouted the task to the `DataAnalyst` agent.
*   **Automated Code Repair**: The "Generator-Critic" loop successfully resolved **70%** of syntax and logic errors within defined iteration limits ($k=3$).
*   **Operational Stability**: The system maintained a **Stability Score of 0.92** over prolonged stress testing, handling OOM exceptions via graceful model degradation (Llama-3 $\rightarrow$ TinyLlama).

## 6.3. User Experience & Visual Analytics
Figure 4 and Figure 5 depict the frontend interface. The Interactive Dashboard provides non-technical users with a "Glass-Box" view of the agentic swarm, visualizing the real-time delegation of tasks (Figure 4) and the resulting data visualizations (Figure 5). This transparency is critical for building user trust in autonomous systems.

[Insert Figure 5 Here]

Figure 5 describes the Multi-agent LLM Analytics platform interface. It enables dataset upload and Natural Language querying, generating visual depictions of the resulting analysis (e.g., pie charts). The User Interface is designed to be aesthetically pleasing with a modern style featuring a gradient background and an attractive layout.

VII. CONCLUSION & FUTURE DIRECTIONS

This work presents the Domain-Flexible Autonomous Data Analysis Agent, a framework that successfully bridges the gap between natural language intent and executable analytical workflows. By orchestrating a swarm of specialized agents through a complexity-aware routing mechanism, the system achieves a 51% reduction in latency compared to monolithic approaches while maintaining high fault tolerance (0.92 stability).

**Future Work**:
Future iterations will focus on three key advancements: 1) **Learning-Based Routing**: Training a classifier on interaction logs to replace static heuristics, potentially improving routing precision by 5-10%; 2) **Out-of-Core Processing**: Integrating Polars/Dask to support datasets exceeding RAM limits; and 3) **Distributed State**: Migrating the `SwarmContext` to Redis to enable fault-tolerant distributed deployments. These enhancements will further solidify the DF-Agent as a robust solution for enterprise-grade autonomous analytics.

##### References

[1] Z. Xi, et al., "The Rise and Potential of Large Language Model Based Agents: A Survey," *arXiv preprint arXiv:2309.07864*, 2023.
[2] L. Wang, et al., "A Survey on Large Language Model based Autonomous Agents," *arXiv preprint arXiv:2308.11432*, 2023.
[3] T. Guo, et al., "Large Language Model based Multi-Agents: A Survey of Progress and Challenges," *arXiv preprint arXiv:2402.01680*, 2024.
[4] W. Chen, et al., "AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors," *ICLR*, 2024.
[5] Q. Wu, et al., "AutoGen: Enabling Next-Gen LLM Applications," *arXiv preprint arXiv:2308.08155*, 2023.
[6] M. Nejjar, et al., "LLMs for Science: Usage for Code Generation and Data Analysis," *arXiv preprint arXiv:2311.16733v4*, 2024.
[7] J. A. Jansen, et al., "Leveraging large language models for data analysis automation," *PLOS ONE*, vol. 20, no. 2, e0317084, 2025.
[8] M. Sun, et al., "From Automation to Autonomy: A Survey on Large Language Models in Scientific Discovery," *arXiv preprint arXiv:2412.14222*, 2024.
[9] C. Chen and T. Stadler, "GenSpectrum Chat: Data Exploration in Public Health Using Large Language Models," *ETH Zurich*, 2023.
[10] Y. Gu, et al., "Large Language Models for Constructing and Optimizing Machine Learning Workflows," *arXiv preprint arXiv:2411.10478*, 2024.
[11] D. Piskala, et al., "OptiRoute: Dynamic LLM Routing and Selection based on User Preferences," *Int. Journal of Computer Applications*, 2024.
[12] J. Talcott, et al., "Universal Model Routing for Efficient LLM Inference (UniRoute)," *arXiv preprint arXiv:2502.08773*, 2025.
[13] Z. Pan, et al., "Route to Reason: Adaptive Routing for LLM and Reasoning Strategy Selection," *arXiv preprint arXiv:2505.19435*, 2025.
[14] A. Heakl, et al., "Dr.LLM: Dynamic Layer Routing in LLMs," *arXiv preprint arXiv:2510.12773*, 2025.
[15] K. Gou, et al., "CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing," *ICLR*, 2024.
[16] A. Madaan, et al., "Self-Refine: Iterative Refinement with Self-Feedback," *NeurIPS*, 2023.
[17] S. Dhuliawala, et al., "Chain-of-Verification Reduces Hallucination in Large Language Models," *arXiv:2309.11495*, 2023.
[18] A. Kumar, et al., "Training Language Models to Self-Correct via Reinforcement Learning (SCoRe)," *arXiv preprint arXiv:2409.12917*, 2024.
[19] Y. Dong, et al., "A Survey on Code Generation with LLM-based Agents," *arXiv preprint arXiv:2508.00083*, 2025.
[20] S. Bistarelli, et al., "Usage of Large Language Model for Code Generation Tasks: A Review," *SN Computer Science*, vol. 6, 2025.
[21] J.S. Park, et al., "Generative Agents: Interactive Simulacra of Human Behavior," *UIST*, 2023.
[22] C. Zhang, et al., "MindAgent: Emergent Gaming Interaction," *arXiv:2309.09971*, 2023.
[23] A. Thirunagalingam, "Bias Detection and Mitigation in Data Pipelines," *AVE Trends in Intelligent Computing Systems*, 2024.
[24] F. Chiarello, et al., "Future applications of generative large language models," *Technovation*, vol. 133, 2024.
``
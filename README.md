# Dynamic Demonstration Retrieval and Cognitive Understanding for Emotional Support Conversation

![DRCU](https://github.com/Bat-Reality/DDRCU/assets/56718188/491bb39f-9c96-43cd-bf98-181391e9a43b)

### Abstract
Emotional Support Conversation (ESC) systems are pivotal in providing empathetic interactions, aiding users through negative emotional states by understanding and addressing their unique experiences. In this paper, we tackle two key challenges in ESC: enhancing contextually relevant and empathetic response generation through dynamic demonstration retrieval, and advancing cognitive understanding to grasp implicit mental states comprehensively. We introduce Dynamic Demonstration Retrieval and Cognitive-Aspect Situation Understanding (\ourwork), a novel approach that synergizes these elements to improve the quality of support provided in ESCs. By leveraging in-context learning and persona information, we introduce an innovative retrieval mechanism that selects informative and personalized demonstration pairs. We also propose a cognitive understanding module that utilizes four cognitive relationships from the ATMOIC knowledge source to deepen situational awareness of help-seekers' mental states. Our supportive decoder integrates information from diverse knowledge sources, underpinning response generation that is both empathetic and cognitively aware. The efficacy of \ourwork is demonstrated through extensive automatic and human evaluations, revealing substantial improvements over existing state-of-the-art models, with up to 12\% enhancement in performance metrics. Our approach not only propels the field of ESC systems forward but also provides a resource for ongoing research, with our codes made available for public access.

### D$^2$RCU
Our code (in codes) mainly references [https://github.com/chengjl19/PAL/tree/main](https://github.com/chengjl19/PAL/tree/main)

### Model training

You should first download the [BlenderBot-small](https://huggingface.co/facebook/blenderbot_small-90M) model and put the `pytorch_model.bin` file in `Blenderbot_small-90M`.

You should then download the [dpr-reader-single-nq-base](https://huggingface.co/facebook/dpr-reader-single-nq-base) model and put the `pytorch_model.bin` file in `DPR`.

Then run `RUN/prepare_strat.sh`

And run `RUN/train_strat.sh`

### Model inference

run `RUN/infer_strat.sh`

### Model interact

run `RUN/interact_strat.sh`

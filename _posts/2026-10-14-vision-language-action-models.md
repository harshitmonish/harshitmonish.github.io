---
title: "From Seeing to Doing: Vision-Language and Vision-Language-Action Models"
date: 2026-10-14
permalink: /posts/2026/10/vision-language-action-models/
tags:
  - Artificial Intelligence
  - Robotics
  - Vision-Language Models
  - Vision-Language-Action Models
  - Machine Learning
---

# From Seeing to Doing: VLMs and VLA Models

Suppose I ask a robot to put the red mug next to the kettle. It has to find both objects, figure out where the mug is, move its gripper to it, pick it up, and set it down without hitting anything. A model that can describe the scene has only done the first part of the job.

That’s the difference I want to unpack here. Vision-language models (VLMs) connect images with words. Vision-language-action (VLA) models use visual and language input to choose actions for a robot. The names are similar, but making a prediction about an image and moving something in the real world are very different tasks.

## What does a VLM do?

A VLM is trained to connect visual input with language. It may answer questions about an image, produce a caption, locate a named object, or compare two scenes. A common design combines an image encoder with a language model. A projection or connector maps visual features into a form the language model can use alongside text tokens.

<figure>
<svg viewBox="0 0 900 260" role="img" aria-labelledby="vlm-title" xmlns="http://www.w3.org/2000/svg">
  <title id="vlm-title">Vision-language model processing an image and question to produce a grounded text answer</title>
  <defs><marker id="vlm-arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#52647a"/></marker></defs>
  <rect x="22" y="53" width="150" height="66" rx="12" fill="#e8f1ff" stroke="#4776aa" stroke-width="2"/><text x="97" y="82" text-anchor="middle" font-family="sans-serif" font-size="17">Image</text><text x="97" y="104" text-anchor="middle" font-family="sans-serif" font-size="13">pixels / patches</text>
  <rect x="22" y="151" width="150" height="66" rx="12" fill="#fff3df" stroke="#bd8740" stroke-width="2"/><text x="97" y="180" text-anchor="middle" font-family="sans-serif" font-size="17">Question</text><text x="97" y="202" text-anchor="middle" font-family="sans-serif" font-size="13">“What is on the table?”</text>
  <rect x="235" y="53" width="170" height="66" rx="12" fill="#e8f1ff" stroke="#4776aa" stroke-width="2"/><text x="320" y="82" text-anchor="middle" font-family="sans-serif" font-size="17">Vision encoder</text><text x="320" y="104" text-anchor="middle" font-family="sans-serif" font-size="13">visual features</text>
  <rect x="235" y="151" width="170" height="66" rx="12" fill="#fff3df" stroke="#bd8740" stroke-width="2"/><text x="320" y="180" text-anchor="middle" font-family="sans-serif" font-size="17">Text tokens</text><text x="320" y="202" text-anchor="middle" font-family="sans-serif" font-size="13">language features</text>
  <rect x="485" y="91" width="170" height="86" rx="12" fill="#f1eaff" stroke="#8062a8" stroke-width="2"/><text x="570" y="126" text-anchor="middle" font-family="sans-serif" font-size="18">Fusion +</text><text x="570" y="151" text-anchor="middle" font-family="sans-serif" font-size="18">language model</text>
  <rect x="735" y="91" width="145" height="86" rx="12" fill="#e6f5e9" stroke="#4b8b5b" stroke-width="2"/><text x="807" y="126" text-anchor="middle" font-family="sans-serif" font-size="17">Text answer</text><text x="807" y="151" text-anchor="middle" font-family="sans-serif" font-size="13">“A mug and kettle.”</text>
  <path d="M172 86 H235" stroke="#52647a" stroke-width="2" marker-end="url(#vlm-arrow)"/><path d="M172 184 H235" stroke="#52647a" stroke-width="2" marker-end="url(#vlm-arrow)"/><path d="M405 86 C445 86 447 111 485 119" fill="none" stroke="#52647a" stroke-width="2" marker-end="url(#vlm-arrow)"/><path d="M405 184 C445 184 447 157 485 149" fill="none" stroke="#52647a" stroke-width="2" marker-end="url(#vlm-arrow)"/><path d="M655 134 H735" stroke="#52647a" stroke-width="2" marker-end="url(#vlm-arrow)"/>
</svg>
<figcaption align="center"><b>A typical VLM path: turn images and words into a shared representation, then generate a language response.</b></figcaption>
</figure>

There isn’t one standard VLM design. Some take one image; others split a high-resolution image into crops or look at several video frames. Many answer in text, while some can also return things like object coordinates or masks.

The useful question is: what does the model output? A VLM usually gives you information *about* the image. It might say where the mug is or describe how to pick it up. Neither answer tells the arm which direction to move.

## From describing to acting

A robot policy must map observations and a task instruction to actions that its particular body can execute. The observation might include camera images, joint positions, gripper state, or force readings. The action might specify target positions, velocities, gripper commands, or a short chunk of controls.

VLA models bring vision, language, and action into a common learned policy. In the RT-2 work, for example, robot actions are represented as tokens so a model can be trained on both web-scale vision-language examples and robot demonstrations. Other systems use separate action heads or continuous action representations. The design choice matters: the action format needs to fit the robot, the task, and the control frequency.

<figure>
<svg viewBox="0 0 960 300" role="img" aria-labelledby="vla-title" xmlns="http://www.w3.org/2000/svg">
  <title id="vla-title">Closed-loop vision-language-action robot control</title>
  <defs><marker id="vla-arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#52647a"/></marker></defs>
  <rect x="18" y="45" width="170" height="70" rx="12" fill="#e8f1ff" stroke="#4776aa" stroke-width="2"/><text x="103" y="76" text-anchor="middle" font-family="sans-serif" font-size="17">Sensors</text><text x="103" y="99" text-anchor="middle" font-family="sans-serif" font-size="13">camera + robot state</text>
  <rect x="18" y="184" width="170" height="70" rx="12" fill="#fff3df" stroke="#bd8740" stroke-width="2"/><text x="103" y="215" text-anchor="middle" font-family="sans-serif" font-size="17">Instruction</text><text x="103" y="238" text-anchor="middle" font-family="sans-serif" font-size="13">“Move the mug”</text>
  <rect x="262" y="99" width="205" height="100" rx="14" fill="#f1eaff" stroke="#8062a8" stroke-width="2"/><text x="364" y="137" text-anchor="middle" font-family="sans-serif" font-size="19">VLA policy</text><text x="364" y="164" text-anchor="middle" font-family="sans-serif" font-size="14">perceive + condition + act</text>
  <rect x="548" y="99" width="175" height="100" rx="12" fill="#e6f5e9" stroke="#4b8b5b" stroke-width="2"/><text x="635" y="137" text-anchor="middle" font-family="sans-serif" font-size="18">Action chunk</text><text x="635" y="164" text-anchor="middle" font-family="sans-serif" font-size="14">arm motion + gripper</text>
  <rect x="802" y="99" width="140" height="100" rx="12" fill="#fce9e7" stroke="#b36054" stroke-width="2"/><text x="872" y="137" text-anchor="middle" font-family="sans-serif" font-size="18">Robot</text><text x="872" y="164" text-anchor="middle" font-family="sans-serif" font-size="14">acts in the world</text>
  <path d="M188 80 C218 80 225 119 262 127" fill="none" stroke="#52647a" stroke-width="2" marker-end="url(#vla-arrow)"/><path d="M188 219 C220 219 226 179 262 170" fill="none" stroke="#52647a" stroke-width="2" marker-end="url(#vla-arrow)"/><path d="M467 149 H548" stroke="#52647a" stroke-width="2" marker-end="url(#vla-arrow)"/><path d="M723 149 H802" stroke="#52647a" stroke-width="2" marker-end="url(#vla-arrow)"/>
  <path d="M872 199 V275 H103 V254" fill="none" stroke="#52647a" stroke-width="2" stroke-dasharray="7 5" marker-end="url(#vla-arrow)"/><text x="477" y="269" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#52647a">new observation closes the loop; replan as the scene changes</text>
</svg>
<figcaption align="center"><b>A VLA policy is part of a feedback loop. It observes, acts, and observes again.</b></figcaption>
</figure>

The robot shouldn’t plan once and then blindly play back a long sequence. It moves a little, looks again, and adjusts. A long action chunk means fewer model calls, but also more time before it can correct a bad move. Short chunks make it easier to react, though they require more frequent inference.

## Same ingredients, different job

| | VLM | VLA |
|---|---|---|
| Typical inputs | Image or video, text | Visual observations, task text, often robot state |
| Typical output | Caption, answer, description, structured perception | Robot action, action chunk, or control target |
| Main question | “What is happening?” / “Where is the object?” | “What should this robot do next?” |
| Needed grounding | Visual concepts and language | Visual concepts, language, embodiment, and control |
| Common evaluation | Answer quality, localization, grounding | Task success, robustness, safety, and control performance |

A VLM can still be useful in a robot. For example, one component might ask it to identify the mug, then give that result to a conventional motion planner. That system uses a VLM, but the VLM itself isn’t controlling the robot. A VLA’s job is to produce actions from the robot’s observations and the instruction.

<figure>
<svg viewBox="0 0 920 235" role="img" aria-labelledby="compare-title" xmlns="http://www.w3.org/2000/svg">
  <title id="compare-title">Comparison of a VLM answering a question and a VLA controlling a robot</title>
  <defs><marker id="compare-arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#52647a"/></marker></defs>
  <text x="226" y="30" text-anchor="middle" font-family="sans-serif" font-size="19" font-weight="bold">VLM: interpret</text>
  <rect x="28" y="60" width="120" height="55" rx="10" fill="#e8f1ff" stroke="#4776aa"/><text x="88" y="94" text-anchor="middle" font-family="sans-serif" font-size="15">Image + text</text>
  <rect x="305" y="60" width="145" height="55" rx="10" fill="#f1eaff" stroke="#8062a8"/><text x="377" y="94" text-anchor="middle" font-family="sans-serif" font-size="15">VLM</text>
  <rect x="515" y="60" width="145" height="55" rx="10" fill="#e6f5e9" stroke="#4b8b5b"/><text x="587" y="94" text-anchor="middle" font-family="sans-serif" font-size="15">Text / perception</text>
  <path d="M148 87 H305" stroke="#52647a" stroke-width="2" marker-end="url(#compare-arrow)"/><path d="M450 87 H515" stroke="#52647a" stroke-width="2" marker-end="url(#compare-arrow)"/>
  <text x="226" y="153" text-anchor="middle" font-family="sans-serif" font-size="19" font-weight="bold">VLA: control</text>
  <rect x="28" y="175" width="120" height="45" rx="10" fill="#e8f1ff" stroke="#4776aa"/><text x="88" y="203" text-anchor="middle" font-family="sans-serif" font-size="14">Sensors + goal</text>
  <rect x="305" y="175" width="145" height="45" rx="10" fill="#f1eaff" stroke="#8062a8"/><text x="377" y="203" text-anchor="middle" font-family="sans-serif" font-size="14">VLA policy</text>
  <rect x="515" y="175" width="145" height="45" rx="10" fill="#e6f5e9" stroke="#4b8b5b"/><text x="587" y="203" text-anchor="middle" font-family="sans-serif" font-size="14">Robot action</text>
  <rect x="740" y="175" width="145" height="45" rx="10" fill="#fce9e7" stroke="#b36054"/><text x="812" y="203" text-anchor="middle" font-family="sans-serif" font-size="14">World changes</text>
  <path d="M148 197 H305" stroke="#52647a" stroke-width="2" marker-end="url(#compare-arrow)"/><path d="M450 197 H515" stroke="#52647a" stroke-width="2" marker-end="url(#compare-arrow)"/><path d="M660 197 H740" stroke="#52647a" stroke-width="2" marker-end="url(#compare-arrow)"/>
</svg>
<figcaption align="center"><b>Both connect vision and language. A VLA also has to connect its output to physical effectors.</b></figcaption>
</figure>

## Where do the actions come from?

Usually, from demonstrations. Someone operates the robot, and the training data records what it saw, what task it was doing, and how it moved. The model is trained on many of these trajectories. The more variety in tasks, objects, viewpoints, and robots, the better chance it has of handling a situation beyond the exact demonstrations.

Images and captions from the web can teach a model that a whisk is a kitchen tool, but they don’t tell it how far to move a particular arm to pick one up. Robot demonstrations supply that connection. Some VLA training mixes both kinds of data; RT-2 is a well-known example.

Actions can be encoded in several ways:

* **Discrete action tokens:** quantize dimensions such as position, rotation, or gripper state into bins, then predict token sequences. This fits naturally into an autoregressive language model, though quantization and token-by-token generation can be limiting.
* **Continuous action heads:** predict real-valued controls directly, often with a specialized policy head.
* **Action chunks or trajectories:** generate a short sequence of future controls together. This can make execution smoother and amortize inference, but errors may persist until the next replanning step.
* **Generative action policies:** model a distribution over plausible motions rather than one average action, useful when several valid ways to complete a task exist.

The choice depends on the job. A robot stacking blocks slowly has different timing needs from one handling objects with both hands. There’s no single action format that fits all of them.

## Why people are interested in them

### Language can specify the goal flexibly

Language is a convenient way to describe a goal. A model trained on many objects and instructions might handle a new combination without needing a separate task ID or a new model for every object. “Might” is doing some work here: it depends on the training data and still has to be checked on the robot.

### Shared policies can reuse experience

The other appeal is sharing experience between tasks. If the model has learned to grasp handles on several objects, that skill may carry over to a new container or tool. That’s the hope behind training one general policy instead of a separate policy for every task.

### They connect semantic decisions to motor output

The catch is that a single learned policy can be harder to inspect than a system with explicit perception, planning, and control modules. And the VLA label doesn’t guarantee generalization. It depends on the data, the robot, and what was actually tested.

## What makes this difficult in practice?

### Robot data is expensive and uneven

Images and text are easy to find online. Robot trajectories are not. Each one depends on a particular robot, camera setup, workspace, and controller. A dataset can list lots of different tasks and still show the model the same few objects, viewpoints, and motion styles over and over.

The model also has to connect its movement to what happens next. A camera may not show depth or hidden geometry; it may be hard to tell whether a grasp succeeded until the robot tries to lift the object. A model can learn to imitate demonstrations without learning a rule that holds up in a new situation.

### Embodiment matters

An action only makes sense for a particular robot. One arm may reach a spot that another can’t; grippers, calibration, and joint limits differ too. Getting a policy to work across robots means accounting for those differences, not just reusing the same command.

### Good language answers do not prove good control

It might correctly answer that the cup is left of the plate and still miss the cup. Picking it up takes accurate motion, good timing, feedback, and some way to recover if the first grasp slips. A strong score on image questions doesn’t tell us whether a robot can do that safely.

### Closed-loop execution has real constraints

Timing matters. If the model takes too long to respond, the scene may have changed by the time the robot acts. A grasp can slip, an object can move, or a person can enter the workspace. A real system needs collision checks, monitoring, an emergency stop, and a way to stop or hand off a task when it’s uncertain.

### Evaluation can hide fragility

One success rate can hide a lot. Maybe the robot works in bright light but fails near a window, or handles cups but not bowls. Tests should include new scenes and objects, repeated attempts, small disturbances, recovery, and response time—not only the examples used in a demo.

## Does the VLA have to do everything?

No. A VLA can suggest the next action while other parts of the robot stack check whether that action is safe and execute it. State estimation, joint limits, collision checking, and low-level motor control don’t disappear just because a learned model is involved.

```text
camera + robot state + instruction
                │
                ▼
       VLM / VLA policy ───────► proposed action
                │                      │
       confidence / status       safety and limit checks
                                       │
                                       ▼
                              low-level controller
                                       │
                                       ▼
                                    robot
                                       │
                          fresh sensor observation ──┐
                                                     └──► replan
```

This split also makes failures easier to track: did the model suggest a bad move, did the controller fail to execute it, or did the sensor miss something? A safety layer won’t fix a poor policy, but it can reject commands that violate basic limits.

For a slow pick-and-place demo, a larger model and frequent replanning may be fine. For a fast task involving contact, latency matters much more, and a dedicated controller may need to handle the quick corrections.

## The short version

The quick distinction: a VLM tells you something about what it sees; a VLA uses what it sees, along with an instruction, to choose what the robot should do next.

```text
VLM: image + words ──► grounded information
VLA: observation + goal ──► action ──► new observation ──► next action
```

That last step is a big one. The model has to work with a particular body, learn from robot data, respond on time, and cope with a world that doesn’t hold still. That’s why a VLA is more than a VLM with a robot arm attached.

## References

* Brohan et al., [“RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control”](https://arxiv.org/abs/2307.15818), 2023.
* Kim et al., [“OpenVLA: An Open-Source Vision-Language-Action Model”](https://arxiv.org/abs/2406.09246), 2024.
* Kawaharazuka et al., [“Vision-Language-Action Models for Robotics: A Review Towards Real-World Applications”](https://arxiv.org/abs/2510.07077), 2025.

---

*The diagrams are simplified illustrations of common patterns; individual models differ.*

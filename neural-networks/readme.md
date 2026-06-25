# Neural Networks — Notes

Quick notes on how a network actually learns. Read top to bottom.

## Why we need activation functions

Stack two linear layers and watch what happens:

```
Layer 1:  y = W₁x + b₁
Layer 2:  z = W₂y + b₂

z = W₂(W₁x + b₁) + b₂
z = (W₂W₁)x + (W₂b₁ + b₂)   →   just z = W'x + b'
```

So 100 linear layers collapse into a single one. No matter how deep you go, you only ever get a straight line. That's the **linearity problem**.

Activation functions fix this by adding a *bend* between layers. The bends stop the layers from collapsing, which is what lets a deep network learn complex, layered patterns. With non-linearity, a network can approximate *any* continuous function — this is the universal approximation idea, and it's the whole reason depth works.

## The common activations

| Function | Range | Notes |
|----------|-------|-------|
| **Sigmoid** | 0 → 1 | Classic S-curve. The old default; good for probability outputs. |
| **Tanh** | -1 → 1 | Zero-centred, usually trains faster than sigmoid. |
| **ReLU** | `max(0, x)` | The breakthrough. Cheap, simple, made deep nets practical. |
| **GELU / Swish** | — | Smooth versions of ReLU. What modern LLMs use. |

## The loss function

Loss is a single number telling us how wrong the model is. Lower loss = better predictions, and **the goal of training is to minimise it**.

For regression we usually use Mean Squared Error:

```
L = (1/n) Σ (prediction − target)²
```

Why squared?
- Makes every error positive (no cancelling out).
- Punishes big mistakes much harder than small ones.

Example — predicting a house price:

```
Predicted: $350,000
Actual:    $400,000
Error:     $50,000
Squared:   2,500,000,000
```

## The training loop

The whole thing is just these four steps, repeated:

1. **Forward pass** — data flows through the network, produces a prediction.
2. **Calculate loss** — measure how far off the prediction is.
3. **Backpropagate** — trace the error backward, work out which weights are to blame.
4. **Update weights** — nudge each weight to reduce future error.

### Backpropagation — assigning blame

The question it answers: *"The prediction was wrong — which weights caused it?"*

The error signal travels backward, output → hidden → input, and each weight gets adjusted in proportion to how much it contributed to the mistake. Big contributor, big correction.

### Gradient descent — finding the valley

Picture yourself blindfolded on a hilly landscape, trying to reach the lowest valley. That landscape is the **loss surface** — every possible error the model can make. The **gradient** is the slope under your feet; it tells you which way is "downhill".

The strategy:
- Feel the slope, take a step downhill (toward lower loss).
- Repeat, step after step.
- Aim for the global minimum — lowest error possible.

### Learning rate

The learning rate is the *size* of each step. Get it wrong and training breaks:

- **Too low** → tiny steps, takes forever to reach the bottom.
- **Just right** → moves down smoothly and settles at the minimum.
- **Too high** → overshoots, bounces around, diverges.

## Scaling up

Same four steps (forward → loss → backprop → update), wildly different scale:

```
XOR network   ~20 parameters
GPT-4         ~1 trillion parameters
```

The mechanics don't change — there's just a lot more of them.

## Key takeaways

1. Neural networks are layers of simple neurons doing weighted sums.
2. Training is the iterative process of adjusting weights to minimise loss.
3. Backpropagation traces the error backward to assign blame to weights.
4. Gradient descent is the strategy for stepping toward lower loss.

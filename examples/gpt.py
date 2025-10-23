context = 64
batch_size = 12

from data.shakespeare import load_shakespeare

data = load_shakespeare(context, batch_size)

train_loader = data["train_loader"]
val_loader = data["val_loader"]
encode = data["encode"]
decode = data["decode"]

# transformer hyperparameters

vocab_size = 65
num_heads = 4
d_embed = 128
d_query = 32
d_value = 32
num_blocks = 4
attention_scale = 1
final_scale = 1

# training hyperparameters

lr = 0.1
beta = 0.95
steps = 2001
log_interval = 10
val_interval = 100
val_iters = 20

from modula.atom import Linear
from modula.bond import SplitIntoHeads, MergeHeads, Rope, AttentionQK, CausalMask, Softmax, ApplyAttentionScores, GeLU

def Attention(num_heads, d_embed, d_query, d_value, attention_scale):
    """Multi-head attention"""

    # For keys, queries, and values we add a heads dimension. For the out projection, we remove heads.
    # Remember modules compose right-to-left, and the order is Linear(d_out, d_in)! And @ means compose.
    Q = SplitIntoHeads(num_heads) @ Linear(num_heads * d_query, d_embed)
    K = SplitIntoHeads(num_heads) @ Linear(num_heads * d_query, d_embed)
    V = SplitIntoHeads(num_heads) @ Linear(num_heads * d_value, d_embed)
    W = Linear(d_embed, num_heads * d_value) @ MergeHeads()

    # Read right-to-left: rotate (Q, K) with RoPE, apply Q @ K.T, mask, softmax (with a scale we can choose).
    AttentionScores = Softmax(attention_scale) @ CausalMask() @ AttentionQK() @ Rope(d_query) @ (Q, K)

    # Read right-to-left: apply attention scores, multiply by 1/3 to fix the sensitivity to 1, project back to d_embed.
    return W @ (1/3 * ApplyAttentionScores()) @ (V, AttentionScores)


from modula.abstract import Identity
from modula.atom import Embed

def GPT(vocab_size, num_heads, d_embed, d_query, d_value, num_blocks, blocks_mass=5, attention_scale=1.0, final_scale=1.0):
    # Set embed to have mass 1. This controls the proportion of feature learning that it contributes to the whole network.
    embed = Embed(d_embed, vocab_size)
    embed.tare()

    # Let's create attention and MLP layers. 
    att = Attention(num_heads, d_embed, d_query, d_value, attention_scale)
    mlp = Linear(d_embed, 4*d_embed) @ GeLU() @ Linear(4*d_embed, d_embed)

    # For our residual connections, L = 2*num_blocks because each block has two residual connections.
    att_block = (1-1/(2*num_blocks)) * Identity() + 1/(2*num_blocks) * att
    mlp_block = (1-1/(2*num_blocks)) * Identity() + 1/(2*num_blocks) * mlp

    # We can use powers of a module to compose it with itself many times!
    blocks = (mlp_block @ att_block) ** num_blocks

    # Set all transformer blocks to have mass 5 (by default).
    # So 5/7 of the change in the network output is due to the blocks,
    # and 2/7 of the change in output is due to the embedding and out projection.
    blocks.tare(absolute=blocks_mass)

    out = final_scale * Linear(vocab_size, d_embed)

    return out @ blocks @ embed


model = GPT(
    vocab_size=vocab_size,
    num_heads=num_heads,
    d_embed=d_embed,
    d_query=d_query,
    d_value=d_value,
    num_blocks=num_blocks,
    attention_scale=attention_scale,
    final_scale=final_scale,
)

model.jit()

print(model)


import jax
import jax.numpy as jnp

def cross_entropy_loss(w, inputs, targets):
    # We use the logsumexp trick for stable cross entropy
    logits = model(inputs, w)  # shape is [batch, seq_len, vocab_size]
    batch_indices = jnp.arange(logits.shape[0])[:, None]  # shape is [batch, 1]
    seq_indices = jnp.arange(logits.shape[1])[None, :]    # shape is [1, seq_len]
    # This indexing selects out logits[b, s, targets[b, s]], which is the target logit
    losses = -logits[batch_indices, seq_indices, targets] + jax.nn.logsumexp(logits, axis=-1)  # shape is [batch, seq_len]
    return losses.mean()

loss_and_grad = jax.jit(jax.value_and_grad(cross_entropy_loss))


key = jax.random.PRNGKey(0)
w = model.initialize(key)

step = 0
momentum = [0 * weight for weight in w]
lr_schedule = lambda step: lr * (steps - step) / steps
for inputs, targets in train_loader:
    loss, grad_w = loss_and_grad(w, inputs, targets)
    momentum = [beta * m + (1 - beta) * g_w for m, g_w in zip(momentum, grad_w)]
    d_w = model.dualize(momentum, target_norm=0.1)
    w = [weight - lr_schedule(step) * d_weight for weight, d_weight in zip(w, d_w)]

    if step % log_interval == 0:
        print(f"Step {step}: loss {loss}")
    
    if step % val_interval == 0:
        val_losses = []
        for val_inputs, val_targets in val_loader:
            loss, _ = loss_and_grad(w, val_inputs, val_targets)
            val_losses.append(loss)
            if len(val_losses) >= val_iters:
                break
        print(f"--> val loss {sum(val_losses)/len(val_losses)}")

    step += 1

    if step >= steps:
        break
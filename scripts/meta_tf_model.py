import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def build_meta_model(F,E,H,Bf,R,A_np,M_rx_np,receptor_targets_np):
    x_struct = keras.Input(shape=(F,), name="struct")
    x_text   = keras.Input(shape=(E,), name="text_embed")
    lr_pos   = keras.Input(shape=(H,Bf), name="lr_pos")
    lr_neg   = keras.Input(shape=(H,Bf), name="lr_neg")
    obs_bin  = keras.Input(shape=(Bf,), name="befund_binary")
    priors   = keras.Input(shape=(H,), name="priors")

    tri = layers.Dense(8)(layers.Concatenate()([x_struct,x_text]))
    tri = layers.Softmax(name="triage_prio")(tri)

    A = tf.constant(A_np, dtype=tf.float32)
    Z = layers.Dense(64, activation="relu")(layers.Concatenate()([x_struct,x_text]))
    for _ in range(2):
        Z = tf.matmul(A, Z)
        Z = layers.Dense(64, activation="relu")(Z)
    graph_embed = layers.Dense(64, activation="relu")(Z)

    def bayes_update(args):
        pri, lp, ln, ob = args
        odds = pri/(1.-pri+1e-9)
        lr   = tf.where(tf.expand_dims(ob,1)>0.5, lp, ln)
        lr   = tf.reduce_prod(lr+1e-9, axis=2)
        odds_post = odds*lr
        post = odds_post/(1.+odds_post)
        post = post/(tf.reduce_sum(post, axis=1, keepdims=True)+1e-9)
        return post

    post = layers.Lambda(bayes_update, name="bayes")([priors, lr_pos, lr_neg, obs_bin])

    fuse = layers.Concatenate()([x_struct,x_text,graph_embed,post])
    fuse = layers.Dense(256, activation="relu")(fuse)
    fuse = layers.Dropout(0.2)(fuse)

    dx_prob  = layers.Dense(H, activation="softmax", name="dx_prob")(fuse)
    severity = layers.Dense(3, activation="softmax", name="severity")(fuse)
    ae_risk  = layers.Dense(R, activation="sigmoid", name="adverse_events")(fuse)
    net_ben  = layers.Dense(1, activation="linear", name="net_benefit")(fuse)

    M_rx = tf.constant(M_rx_np, dtype=tf.float32)
    receptor_target = tf.constant(receptor_targets_np, tf.float32)
    need = tf.matmul(dx_prob, receptor_target)
    match = tf.matmul(need, M_rx, transpose_b=True)
    policy_score = match
    rx_rec = layers.Activation("softmax", name="rx_recommendation")(policy_score)

    model = keras.Model(
        inputs=[x_struct,x_text,lr_pos,lr_neg,obs_bin,priors],
        outputs=[dx_prob,severity,ae_risk,net_ben,rx_rec,tri,post]
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss={"dx_prob":"kld","severity":"categorical_crossentropy",
                        "adverse_events":"binary_crossentropy","net_benefit":"mse"})
    return model

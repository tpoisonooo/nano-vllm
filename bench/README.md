

```text
================================================================================
Layer Name                               Count    Total (ms)   Avg (ms)    
================================================================================
model.layers.6.self_attn                 256      392.228      1.532        
model.layers.5.mlp                       256      55.736       0.218          
model.layers.1.post_attention_layernorm  256      35.532       0.139       
model.layers.1.input_layernorm           256      32.183       0.126         
================================================================================


================================================================================
Layer Name                               Count    Total (ms)   Avg (ms)    
================================================================================
model.layers.27.self_attn.rotary_emb     7168     1619.966     0.226       
model.layers.24.self_attn.attn           256      35.793       0.140       
model.layers.9.self_attn.q_norm          256      35.397       0.138       
model.layers.1.mlp.act_fn                256      30.823       0.120       
model.layers.1.self_attn.k_norm          256      29.925       0.117      
model.layers.26.self_attn.o_proj         256      9.881        0.039       
model.layers.1.mlp.down_proj             256      9.481        0.037        
model.layers.1.mlp.gate_up_proj          256      8.731        0.034    
model.layers.1.self_attn.qkv_proj        256      8.265        0.032     
================================================================================
```

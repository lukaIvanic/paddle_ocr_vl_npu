if(BUILD_OPEN_PROJECT)
    set(
        paddle_gqa_incre_flash_attention_aiv_depends
        attention/prompt_flash_attention
        CACHE INTERNAL
        "Dependencies for PaddleGqaIncreFlashAttentionAiv"
    )
    target_sources(op_host_aclnnInner PRIVATE
        paddle_gqa_incre_flash_attention_aiv_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME PaddleGqaIncreFlashAttentionAiv
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -mllvm -cce-vf-remove-membar=false
            -Werror
)

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources_with_soc(
        OPTYPE paddle_gqa_incre_flash_attention_aiv
        ACLNNTYPE aclnn_inner
    )
endif()


/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0.
 *
 * CANN 9.x exports this public registry API from libopp_registry.so but does
 * not install op_impl_registry_base.h. Keep the matching public declaration
 * here so the extension can replace one broken infer callback without
 * replacing the operator tiler or kernel implementation.
 */

#ifndef UNIREC_OP_IMPL_REGISTRY_BASE_COMPAT_H_
#define UNIREC_OP_IMPL_REGISTRY_BASE_COMPAT_H_

#include <map>

#include "register/op_impl_kernel_registry.h"

namespace gert {

struct OpImplRegistryBase : public OpImplKernelRegistry {
    virtual ~OpImplRegistryBase() = default;
    virtual const OpImplFunctions *GetOpImpl(
        const ge::char_t *op_type) const = 0;
    virtual const OpImplRegisterV2::PrivateAttrList &GetPrivateAttrs(
        const ge::char_t *op_type) const = 0;
};

class OpImplRegistry : public OpImplRegistryBase {
 public:
    static OpImplRegistry &GetInstance();
    OpImplFunctionsV2 &CreateOrGetOpImpl(const ge::char_t *op_type);
    const OpImplFunctionsV2 *GetOpImpl(
        const ge::char_t *op_type) const override;
    const OpImplRegisterV2::PrivateAttrList &GetPrivateAttrs(
        const ge::char_t *op_type) const override;
    const std::map<OpImplRegisterV2::OpType, OpImplFunctionsV2> &
        GetAllTypesToImpl() const;
    std::map<OpImplRegisterV2::OpType, OpImplFunctionsV2> &
        GetAllTypesToImpl();

 private:
    std::map<OpImplRegisterV2::OpType, OpImplFunctionsV2> types_to_impl_;
    uint8_t reserved_[40] = {0U};
};

} // namespace gert

#endif

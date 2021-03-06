// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <iterator>
#include "itt.hpp"

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/builder/norm.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/grn.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_SUPPRESS_DEPRECATED_START

NGRAPH_RTTI_DEFINITION(op::GRN, "GRN", 0, op::util::FusedOp);

op::GRN::GRN()
    : FusedOp()
{
}

op::GRN::GRN(const Output<Node>& data, float bias)
    : FusedOp({data})
    , m_bias(bias)
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::GRN::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_GRN_visit_attributes);
    visitor.on_attribute("bias", m_bias);
    return true;
}

void op::GRN::pre_validate_and_infer_types()
{
    const auto& data_pshape = get_input_partial_shape(0);

    if (data_pshape.is_static())
    {
        const Shape& data_shape{data_pshape.to_shape()};

        // Input data must be 2, 3 or 4D tensor.
        NODE_VALIDATION_CHECK(this,
                              (data_shape.size() >= 2 && data_shape.size() <= 4),
                              "Input tensor rank must be 2, 3 or 4 dimensional (actual input "
                              "shape: ",
                              data_shape,
                              ").");
    }
}

OutputVector op::GRN::decompose_op() const
{
    Output<Node> data{input_value(0)};
    const Shape& input_shape{data.get_shape()};

    // Reshape to 4D tensor.
    if (input_shape.size() != 4)
    {
        Shape data_shape(4 - input_shape.size(), 1);
        copy(begin(input_shape), end(input_shape), back_inserter(data_shape));
        data = builder::opset1::reshape(data, data_shape);
    }

    const auto axis_set_const = op::Constant::create(element::i64, {}, {1});
    // Calculate l2 norm across channels.
    shared_ptr<Node> norm = builder::opset1::l2_norm(data, axis_set_const, m_bias);
    // Get back reduced axis.
    data = std::make_shared<op::v1::Divide>(
        data, builder::opset1::make_broadcast(norm, data.get_shape(), AxisSet{1}));

    // get back original input tensor rank
    if (input_shape.size() != 4)
    {
        data = builder::opset1::reshape(data, input_shape);
    }

    return OutputVector{data};
}

shared_ptr<Node> op::GRN::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_GRN_clone_with_new_inputs);
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<GRN>(new_args.at(0), m_bias);
}

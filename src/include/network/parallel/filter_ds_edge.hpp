#pragma once

#include "edge.hpp"
#include "edges_fwd.hpp"
#include "nodes.hpp"

#include "../../convolution/convolution.hpp"
#include "../filter.hpp"

namespace znn { namespace v4 { namespace parallel_network {

class filter_ds_edge: public edge
{
private:
    vec3i    filter_stride;
    vec3i    repeat_;
    filter & filter_;

    ccube_p<real> last_input;
    ccube_p<real> pending_input;

    //task_manager::task_handle pending_ = 0;

    std::mutex m;

private:
    void do_forward( ccube_p<real> const & f )
    {
        last_input = f;
        out_nodes->forward(out_num,
                           convolve_sparse(*f, filter_.W(), filter_stride));
    }

    void do_update( ccube_p<real> const & g )
    {
        auto dEdW = convolve_sparse_flipped(*last_input, *g, filter_stride);
        filter_.update(*dEdW);
        flatten(filter_.W(), repeat_);

        {
            guard gg(m);
            last_input.reset();
            if ( pending_input )
            {
                do_forward(std::move(pending_input));
            }
        }
    }

public:
    filter_ds_edge( nodes * in,
                    size_t inn,
                    nodes * out,
                    size_t outn,
                    task_manager & tm,
                    vec3i const & stride,
                    vec3i const & repeat,
                    filter & f )
        : edge(in,inn,out,outn,tm),
          filter_stride(stride),
          repeat_(repeat),
          filter_(f)
    {
        in->attach_out_edge(inn,this);
        out->attach_in_edge(outn,this);
        flatten(filter_.W(), repeat_);
    }

    void forward( ccube_p<real> const & f ) override
    {
        {
            guard gg(m);
            if ( !last_input )
            {
                manager.schedule(this->fwd_priority(),
                                 &filter_ds_edge::do_forward, this, f);
            }
            else
            {
                pending_input = f;
            }
        }
    }

    void backward( ccube_p<real> const & g )
    {
        guard gg(m);
        ZI_ASSERT(last_input);
        in_nodes->backward(in_num,
                           convolve_sparse_inverse(*g,
                                                   filter_.W(),
                                                   filter_stride));

        manager.schedule( this->fwd_priority() + 512,
                          &filter_ds_edge::do_update, this, g );
    }

};


}}} // namespace znn::v4::parallel_network

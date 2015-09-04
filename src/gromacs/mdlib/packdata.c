/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team.
 * Copyright (c) 2013,2014,2015, by the GROMACS development team, led by
 * Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
 * and including many others, as listed in the AUTHORS file in the
 * top-level source directory and at http://www.gromacs.org.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */
#include <stdlib.h>
#include <string.h>
#include "packdata.h"
#include "gromacs/utility/smalloc.h"

__declspec(target(mic))
void *roundup_ptr(void *addr)
{
    return (void *)(((size_t)((char *)addr + PACK_BUFFER_ALIGN - 1)) & (~(PACK_BUFFER_ALIGN - 1)));
}

__declspec(target(mic))
size_t roundup_size(size_t size)
{
    return (size + PACK_BUFFER_ALIGN - 1) & (~(PACK_BUFFER_ALIGN - 1));
}

__declspec(target(mic))
size_t compute_header_size(int num_buffers)
{
    return num_buffers * 2 * sizeof(size_t);
}

void packdata(void *packet, packet_buffer *buffers, int num_buffers)
{
    int   i;
    char *header_ptr = (char *)packet;
    char *data_ptr   = (char *)roundup_ptr(header_ptr + compute_header_size(num_buffers));
    for (i = 0; i < num_buffers; i++)
    {
        memcpy(header_ptr, &(buffers[i].s), sizeof(size_t));
        header_ptr += sizeof(size_t);
        size_t ptr_offset = (size_t)(data_ptr - (char *)packet);
        memcpy(header_ptr, &ptr_offset, sizeof(size_t));
        header_ptr += sizeof(size_t);
        memcpy(data_ptr, buffers[i].p, buffers[i].s);
        data_ptr = roundup_ptr(data_ptr + buffers[i].s);
    }
}

void unpackdata(void *packet, void **buffers, int num_buffers)
{
    int   i;
    char *header_ptr = (char *)packet;
    for (i = 0; i < num_buffers; i++)
    {
        size_t size = *(size_t *)header_ptr;
        header_ptr += sizeof(size_t);
        size_t offset = *(size_t *)header_ptr;
        header_ptr += sizeof(size_t);
        memcpy(buffers[i], (char *)packet + offset, size);
    }
}

size_t compute_required_size(packet_buffer *buffers, int num_buffers)
{
    int    i;
    size_t size = compute_header_size(num_buffers) + PACK_BUFFER_ALIGN;
    for (i = 0; i < num_buffers-1; i++)
    {
        size += roundup_size(buffers[i].s);
    }
    size += buffers[num_buffers-1].s;
    return size;
}

packet_buffer get_buffer(void *packet, int buffer_num)
{
    int           i;
    char         *ptr = (char *)packet;
    packet_buffer buf;
    for (i = 0; i < buffer_num; i++)
    {
        ptr += 2*sizeof(size_t);
    }
    buf.s = *(size_t *)ptr;
    ptr  += sizeof(size_t);
    buf.p = (char *)packet + *(size_t *)ptr;
    return buf;
}

void create_packet_iter(void *packet, packet_iter *iter)
{
    iter->packet = (char *)packet;
    iter->ptr    = (char *)packet;
}

void *value(packet_iter *iter)
{
    return (void *)(iter->packet + (*(size_t *)(iter->ptr + sizeof(size_t))));
}

size_t size(packet_iter *iter)
{
    return *(size_t *)(iter->ptr);
}

void *next(packet_iter *iter)
{
    void *oldval = value(iter);
    iter->ptr += 2*sizeof(size_t);
    return oldval;
}

void *anext(packet_iter *iter, int multiplier)
{
    void  *buffer;
    size_t len = size(iter);
    snew_aligned(buffer, len*multiplier, 64);
    memcpy(buffer, next(iter), len);
    return buffer;
}

void cnext(packet_iter *iter, void *buffer)
{
    size_t bsize = size(iter);
    memcpy(buffer, next(iter), bsize);
}

/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team.
 * Copyright (c) 2013,2014, by the GROMACS development team, led by
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
#define PACK_BUFFER_ALIGN 64
#define size_t unsigned long

typedef struct packet_buffer_struct
{
	void   *p;
	size_t s;
} packet_buffer;

// Packet-level operations
__declspec(target(mic)) void packdata  (void *packet, packet_buffer *buffers, int num_buffers);

__declspec(target(mic)) void unpackdata(void *packet, void **buffers,         int num_buffers);

__declspec(target(mic)) size_t compute_required_size(packet_buffer *buffers, int num_buffers);

__declspec(target(mic)) packet_buffer get_buffer(void *packet, int buffer_num);

// Buffer-level operations
__declspec(target(mic))
typedef struct packet_iter_struct
{
	void *packet;
	void *ptr;
} packet_iter;

__declspec(target(mic)) void create_packet_iter(void *packet, packet_iter *iter);

__declspec(target(mic)) void *value(packet_iter *iter);

__declspec(target(mic))	size_t size(packet_iter *iter);

// Return pointer to current buffer and advance to next buffer
__declspec(target(mic))	void *next(packet_iter *iter);

// Same as "next" except allocates a new buffer and copies the current buffer's contents to it
// instead of returning a pointer to the buffer inside the packet.
__declspec(target(mic)) void *anext(packet_iter *iter);

// Possible future extension "cnext" - copy contents to an existing buffer.


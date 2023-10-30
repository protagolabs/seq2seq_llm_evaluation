"""
This script is used to deduplicate the samples of the Newsela text simplification dataset.
For an explanation of why we did this, see Appendix A of our paper.
"""
import random


if __name__ == '__main__':
    with open('data/newsela-auto/newsela-auto/ACL2020/valid.src', 'r') as f:
        valid_src = f.readlines()
        valid_src = [i.strip("\n") for i in valid_src]

    with open('data/newsela-auto/newsela-auto/ACL2020/valid.dst', 'r') as f:
        valid_dst = f.readlines()
        valid_dst = [i.strip("\n") for i in valid_dst]

    assert len(valid_src) == len(valid_dst)

    src_dedup = []
    dst_dedup = []

    while valid_dst:  # not empty
        for i, (src, dst) in enumerate(zip(valid_src, valid_dst)):
            assert len(valid_src) == len(valid_dst)
            if len(valid_dst) > 1:
                if valid_src[i + 1] != src and valid_dst[i + 1] != dst:
                    src_dedup.append(valid_src.pop(i))
                    dst_dedup.append(valid_dst.pop(i))
                    break
                elif valid_src[i + 1] == src and valid_dst[i + 1] == dst:
                    valid_src.pop(i)
                    valid_dst.pop(i)
                    break
                elif valid_src[i + 1] == src and valid_dst[i + 1] != dst:
                    valid_src.pop(i)
                    valid_dst[i + 1] = dst + ' ' + valid_dst[i + 1]
                    valid_dst.pop(i)
                    break
                elif valid_src[i + 1] != src and valid_dst[i + 1] == dst:
                    valid_src[i + 1] = src + ' ' + valid_src[i + 1]
                    valid_src.pop(i)
                    valid_dst.pop(i)
                    break
            else:
                src_dedup.append(valid_src.pop(i))
                dst_dedup.append(valid_dst.pop(i))

    with open('data/newsela-auto/newsela-auto/ACL2020/valid_dedup.src', 'w') as f:
        f.writelines([i + "\n" for i in src_dedup])

    with open('data/newsela-auto/newsela-auto/ACL2020/valid_dedup.dst', 'w') as f:
        f.writelines([i + "\n" for i in dst_dedup])

    NUM_SAMPLES = 3000  # pick 3000 samples to be used for automatic metric evaluation
    indices = random.sample(range(1, len(src_dedup)), NUM_SAMPLES)
    src_dedup_sample = [src_dedup[i] for i in indices]
    dst_dedup_sample = [dst_dedup[i] for i in indices]

    with open('data/newsela-auto/newsela-auto/ACL2020/valid_dedup_sample.src', 'w') as f:
        f.writelines([i + "\n" for i in src_dedup_sample])

    with open('data/newsela-auto/newsela-auto/ACL2020/valid_dedup_sample.dst', 'w') as f:
        f.writelines([i + "\n" for i in dst_dedup_sample])

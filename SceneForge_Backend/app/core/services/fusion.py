def fuse_frames(frames_data):
    # Example fusion: merge all point lists into one unique set
    fused_points = set()
    for data in frames_data:
        fused_points.update(data)
    return list(fused_points)
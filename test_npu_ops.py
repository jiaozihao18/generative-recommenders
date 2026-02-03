"""
Minimal check: 各 NPU/GR 相关算子在当前环境是否可用。
在项目根目录或 PYTHONPATH 含 generative_recommenders 的目录下运行：
  python test_npu_ops.py
"""
import os
import sys

def test(name, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        print(f"[OK] {name}")
        return True
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
        return False

def main():
    ok = True

    # 1) fbgemm（通常由 fbgemm_gpu 注册）
    try:
        import fbgemm_gpu  # noqa: F401
    except Exception as e:
        print(f"[SKIP] fbgemm_gpu not available: {e}")
        return
    if not hasattr(__import__("torch").ops, "fbgemm"):
        print("[FAIL] torch.ops.fbgemm not found after importing fbgemm_gpu")
        ok = False
    else:
        torch = __import__("torch")
        # 设备：NPU 则用 npu，否则 cuda/cpu
        if hasattr(torch, "npu") and torch.npu.is_available():
            dev = "npu:0"
        elif torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        lengths = torch.tensor([2, 3], dtype=torch.int64, device=dev)
        x_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        if not test("torch.ops.fbgemm.asynchronous_complete_cumsum", lambda: (x_offsets,)):
            ok = False
        B, N, D = 2, 5, 4
        x = torch.randn(B, N, D, device=dev)
        try:
            out = torch.ops.fbgemm.dense_to_jagged(x, [x_offsets])
            test("torch.ops.fbgemm.dense_to_jagged (2 args)", lambda: None)
        except TypeError:
            try:
                out = torch.ops.fbgemm.dense_to_jagged(x, [x_offsets], 5)
                test("torch.ops.fbgemm.dense_to_jagged (3 args)", lambda: None)
            except Exception as e:
                print(f"[FAIL] torch.ops.fbgemm.dense_to_jagged: {e}")
                ok = False
        padded = torch.ops.fbgemm.jagged_to_padded_dense(out[0], [x_offsets], [N])
        if not test("torch.ops.fbgemm.jagged_to_padded_dense", lambda: (padded,)):
            ok = False

    # 2) mxrec（NPU 自定义 .so，USE_NPU_HSTU 时用）
    if os.environ.get("USE_NPU_HSTU", "0") == "1":
        try:
            torch = __import__("torch")
            torch.ops.load_library("/home/torch_ops/libhstu_dense_ops.so")
            if not hasattr(torch.ops, "mxrec"):
                print("[FAIL] torch.ops.mxrec not found after loading libhstu_dense_ops.so")
                ok = False
            else:
                # 仅检查符号存在，实际 forward 需要和训练里一致的 q,k,v 等
                if not test("torch.ops.mxrec.hstu_dense (symbol)", lambda: getattr(torch.ops.mxrec, "hstu_dense")):
                    ok = False
                if not test("torch.ops.mxrec.hstu_dense_backward (symbol)", lambda: getattr(torch.ops.mxrec, "hstu_dense_backward")):
                    ok = False
        except Exception as e:
            print(f"[SKIP/FAIL] mxrec (libhstu_dense_ops.so): {e}")
            ok = False
    else:
        print("[SKIP] USE_NPU_HSTU!=1, skip mxrec")

    # 3) torch_npu（仅 ENABLE_RAB=1 时需要）
    if os.environ.get("ENABLE_RAB", "0") == "1":
        try:
            import torch_npu
            torch = __import__("torch")
            x = torch.randn(10, 8, device="npu:0")
            idx = torch.randint(0, 10, (5,), device="npu:0")
            y = torch_npu.gather_for_rank1(x, index=idx)
            if not test("torch_npu.gather_for_rank1", lambda: None):
                ok = False
            # backward 一般要放在 autograd 里测，这里只检查存在
            if not test("torch_npu.index_select_for_rank1_backward (symbol)", lambda: getattr(torch_npu, "index_select_for_rank1_backward")):
                ok = False
        except Exception as e:
            print(f"[SKIP/FAIL] torch_npu: {e}")
            ok = False
    else:
        print("[SKIP] ENABLE_RAB!=1, skip torch_npu gather/index_select")

    print("---")
    if ok:
        print("All checked ops passed. 若数据与配置正确，训练有机会跑通。")
    else:
        print("Some checks failed. 需先解决上述 FAIL 再跑训练。")
        sys.exit(1)

if __name__ == "__main__":
    main()